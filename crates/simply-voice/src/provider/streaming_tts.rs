//! Streaming text-to-speech: text deltas in, audio frames out, cancellable.
//!
//! A [`StreamingTts`] session is a channel pair. The caller sends
//! [`TtsInput`]s (text deltas, a `Flush` at the end of each utterance, a
//! `Cancel` to drop everything queued or playing) and receives [`TtsEvent`]s.
//! The audio format is declared once, when the session opens.
//!
//! **The event contract.** Every `Flush` is answered, in order, by exactly one
//! [`TtsEvent::UtteranceDone`] or one [`TtsEvent::Error`], after that
//! utterance's audio. A `Cancel` is answered by [`TtsEvent::Cancelled`]: the
//! utterances it dropped get no answer of their own, and every event received
//! before `Cancelled` belongs to them, so a caller discards audio up to it.
//! The session stays usable after a `Cancel` or an `Error`. Dropping the input
//! sender speaks what is queued (unflushed text included) and then closes the
//! event receiver.
//!
//! [`BatchStreamingTts`] puts any [`TtsProvider`] behind this trait.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;

use super::tts::TtsProvider;
use crate::audio::{Audio, AudioChunk, AudioFormat};

/// What a caller sends into a streaming TTS session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtsInput {
    /// A delta of the current utterance's text. Spaces are not inserted
    /// between deltas: include them.
    Text(String),
    /// The current utterance's text is complete.
    Flush,
    /// Drop every queued and in-flight utterance, unflushed text included.
    /// Answered by [`TtsEvent::Cancelled`]; the session stays usable.
    Cancel,
}

/// What a streaming TTS session sends back.
#[derive(Debug, Clone)]
pub enum TtsEvent {
    /// Audio in the format the session declared when it opened.
    Audio(AudioChunk),
    /// All audio of the oldest unanswered flushed utterance has been sent.
    UtteranceDone,
    /// The oldest unanswered flushed utterance failed; its audio may be
    /// partial. The session stays usable.
    Error(String),
    /// A `Cancel` took effect: everything received before this belongs to
    /// the utterances it dropped.
    Cancelled,
}

/// Settings for one streaming TTS session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamingTtsConfig {
    /// Provider-specific voice id. Empty uses the provider's default.
    pub voice: String,
}

impl StreamingTtsConfig {
    pub fn new(voice: impl Into<String>) -> Self {
        Self { voice: voice.into() }
    }
}

/// The sender and receiver of an open session, with the audio format every
/// [`TtsEvent::Audio`] in it carries.
pub type TtsSession = (AudioFormat, mpsc::Sender<TtsInput>, mpsc::Receiver<TtsEvent>);

/// Text-to-speech that takes text as it is written and returns audio as it is
/// generated. See the module documentation for the event contract.
#[async_trait]
pub trait StreamingTts: Send + Sync {
    /// Open a session. It lives until the input sender is dropped or the
    /// event receiver is.
    async fn open(&self, cfg: StreamingTtsConfig) -> Result<TtsSession>;
}

/// How many inputs and events a session buffers.
pub(crate) const SESSION_CHANNEL: usize = 256;

/// Puts a batch [`TtsProvider`] behind [`StreamingTts`]: text is buffered
/// until `Flush`, then synthesized, sentence by sentence unless turned off, so
/// a long utterance starts after its first sentence. `Cancel` drops the
/// queued text and the in-flight `synthesize` future.
///
/// Every audio the provider returns is converted to the declared format's
/// encoding; a provider answering at another sample rate or channel count
/// fails that utterance (resampling is the caller's).
pub struct BatchStreamingTts<P: ?Sized> {
    provider: Arc<P>,
    format: AudioFormat,
    split_sentences: bool,
}

impl<P: TtsProvider + ?Sized + 'static> BatchStreamingTts<P> {
    /// `format` is what the provider returns, or the same rate in the
    /// encoding the caller wants.
    pub fn new(provider: Arc<P>, format: AudioFormat) -> Self {
        Self { provider, format, split_sentences: true }
    }

    /// Synthesize each flushed utterance whole instead of sentence by sentence.
    pub fn with_sentence_split(mut self, on: bool) -> Self {
        self.split_sentences = on;
        self
    }
}

#[async_trait]
impl<P: TtsProvider + ?Sized + 'static> StreamingTts for BatchStreamingTts<P> {
    async fn open(&self, cfg: StreamingTtsConfig) -> Result<TtsSession> {
        let (in_tx, in_rx) = mpsc::channel(SESSION_CHANNEL);
        let (ev_tx, ev_rx) = mpsc::channel(SESSION_CHANNEL);
        let worker = BatchWorker {
            provider: Arc::clone(&self.provider),
            format: self.format,
            split_sentences: self.split_sentences,
            voice: cfg.voice,
            events: ev_tx,
        };
        tokio::spawn(worker.run(in_rx));
        Ok((self.format, in_tx, ev_rx))
    }
}

type Synth = Pin<Box<dyn Future<Output = Result<Audio>> + Send>>;

struct BatchWorker<P: ?Sized> {
    provider: Arc<P>,
    format: AudioFormat,
    split_sentences: bool,
    voice: String,
    events: mpsc::Sender<TtsEvent>,
}

impl<P: TtsProvider + ?Sized + 'static> BatchWorker<P> {
    fn synth(&self, text: String) -> Synth {
        let provider = Arc::clone(&self.provider);
        let voice = self.voice.clone();
        Box::pin(async move { provider.synthesize(&text, &voice).await })
    }

    fn utterance(&self, text: &str) -> VecDeque<String> {
        if self.split_sentences {
            split_sentences(text).into()
        } else if text.trim().is_empty() {
            VecDeque::new()
        } else {
            VecDeque::from([text.trim().to_string()])
        }
    }

    async fn run(self, mut input: mpsc::Receiver<TtsInput>) {
        let mut text = String::new();
        // Flushed utterances waiting their turn, each as its sentences.
        let mut queue: VecDeque<VecDeque<String>> = VecDeque::new();
        // The utterance being synthesized: its sentences not yet started.
        let mut current: Option<VecDeque<String>> = None;
        let mut inflight: Option<Synth> = None;
        let mut input_open = true;

        loop {
            while inflight.is_none() {
                match current.as_mut() {
                    Some(sentences) => match sentences.pop_front() {
                        Some(s) => inflight = Some(self.synth(s)),
                        None => {
                            current = None;
                            if self.events.send(TtsEvent::UtteranceDone).await.is_err() {
                                return;
                            }
                        }
                    },
                    None => match queue.pop_front() {
                        Some(u) => current = Some(u),
                        None => break,
                    },
                }
            }
            if !input_open && inflight.is_none() {
                return;
            }

            tokio::select! {
                biased;
                inp = input.recv(), if input_open => match inp {
                    Some(TtsInput::Text(t)) => text.push_str(&t),
                    Some(TtsInput::Flush) => queue.push_back(self.utterance(&std::mem::take(&mut text))),
                    Some(TtsInput::Cancel) => {
                        text.clear();
                        queue.clear();
                        current = None;
                        inflight = None; // drops the provider's future
                        if self.events.send(TtsEvent::Cancelled).await.is_err() {
                            return;
                        }
                    }
                    None => {
                        input_open = false;
                        if !text.trim().is_empty() {
                            queue.push_back(self.utterance(&std::mem::take(&mut text)));
                        }
                    }
                },
                res = async { inflight.as_mut().expect("guarded").await }, if inflight.is_some() => {
                    inflight = None;
                    let ev = match res.and_then(|a| into_format(a, self.format)) {
                        Ok(data) => TtsEvent::Audio(AudioChunk::new(data)),
                        Err(e) => {
                            current = None; // the rest of this utterance is skipped
                            TtsEvent::Error(format!("{e:#}"))
                        }
                    };
                    if self.events.send(ev).await.is_err() {
                        return;
                    }
                }
            }
        }
    }
}

/// `audio`'s samples in `format`'s encoding, or an error when its sample rate
/// or channel count differ from `format`'s.
fn into_format(audio: Audio, format: AudioFormat) -> Result<bytes::Bytes> {
    if audio.format.sample_rate != format.sample_rate || audio.format.channels != format.channels {
        anyhow::bail!(
            "the provider returned {} Hz × {} channels, the stream declared {} Hz × {}",
            audio.format.sample_rate,
            audio.format.channels,
            format.sample_rate,
            format.channels,
        );
    }
    Ok(audio.to_encoding(format.encoding))
}

/// Sentences shorter than this are joined to the next one: a synthesis call
/// per "Hi." costs more in latency than it saves.
const MIN_SENTENCE_CHARS: usize = 24;

/// `text` cut after each sentence end (`.`, `!`, `?`, `…` followed by
/// whitespace) and each line break, short pieces joined to the next, each
/// trimmed; nothing for blank text.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut piece = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        piece.push(c);
        let ends = c == '\n'
            || (matches!(c, '.' | '!' | '?' | '…') && chars.peek().is_none_or(|n| n.is_whitespace()));
        if ends && piece.trim().chars().count() >= MIN_SENTENCE_CHARS {
            out.push(piece.trim().to_string());
            piece.clear();
        }
    }
    let rest = piece.trim();
    if !rest.is_empty() {
        out.push(rest.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Voice;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::Duration;

    /// Returns one f32 sample per character at 24 kHz, after `delay`; records
    /// what it was asked; fails text containing "fail"; sets `dropped` when a
    /// call is dropped before it finished.
    #[derive(Default)]
    struct FakeTts {
        said: Mutex<Vec<(String, String)>>,
        delay: Mutex<Duration>,
        dropped: Arc<AtomicBool>,
        rate: Mutex<Option<u32>>,
    }

    struct DropFlag(Arc<AtomicBool>, bool);
    impl Drop for DropFlag {
        fn drop(&mut self) {
            if !self.1 {
                self.0.store(true, Ordering::SeqCst);
            }
        }
    }

    #[async_trait]
    impl TtsProvider for FakeTts {
        async fn synthesize(&self, text: &str, voice: &str) -> Result<Audio> {
            self.said.lock().unwrap().push((text.to_string(), voice.to_string()));
            let mut flag = DropFlag(Arc::clone(&self.dropped), false);
            let delay = *self.delay.lock().unwrap();
            tokio::time::sleep(delay).await;
            flag.1 = true;
            if text.contains("fail") {
                anyhow::bail!("fake failure");
            }
            let rate = self.rate.lock().unwrap().unwrap_or(24_000);
            let bytes: Vec<u8> = text.chars().flat_map(|_| 0.5f32.to_le_bytes()).collect();
            Ok(Audio::from_f32_bytes(bytes, rate))
        }
        async fn stream(&self) -> Result<(mpsc::Sender<String>, mpsc::Receiver<AudioChunk>)> {
            unimplemented!()
        }
        async fn voices(&self) -> Result<Vec<Voice>> {
            Ok(vec![])
        }
    }

    async fn next(rx: &mut mpsc::Receiver<TtsEvent>) -> TtsEvent {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event in time")
            .expect("session open")
    }

    #[test]
    fn sentences_split_at_ends_and_join_short_ones() {
        let s = split_sentences("Hi. This is the first real sentence. And a second one, also long!\nok");
        assert_eq!(
            s,
            vec!["Hi. This is the first real sentence.", "And a second one, also long!", "ok"]
        );
        assert_eq!(split_sentences("Version 1.5 is out"), vec!["Version 1.5 is out"]);
        assert!(split_sentences("  \n ").is_empty());
    }

    #[tokio::test]
    async fn text_is_buffered_until_flush_then_spoken_sentence_by_sentence() {
        let fake = Arc::new(FakeTts::default());
        let tts = BatchStreamingTts::new(Arc::clone(&fake), AudioFormat::f32(24_000));
        let (format, tx, mut rx) = tts.open(StreamingTtsConfig::new("v1")).await.unwrap();
        assert_eq!(format.sample_rate, 24_000);

        let first = "This is the first sentence here. ";
        let second = "And this is the second one.";
        tx.send(TtsInput::Text(first[..10].into())).await.unwrap();
        tx.send(TtsInput::Text(first[10..].into())).await.unwrap();
        tx.send(TtsInput::Text(second.into())).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(fake.said.lock().unwrap().is_empty(), "nothing synthesized before Flush");

        tx.send(TtsInput::Flush).await.unwrap();
        let TtsEvent::Audio(a) = next(&mut rx).await else { panic!("audio") };
        assert_eq!(a.data.len(), 4 * first.trim().len());
        let TtsEvent::Audio(b) = next(&mut rx).await else { panic!("audio") };
        assert_eq!(b.data.len(), 4 * second.len());
        assert!(matches!(next(&mut rx).await, TtsEvent::UtteranceDone));
        let said = fake.said.lock().unwrap().clone();
        assert_eq!(said, vec![(first.trim().into(), "v1".into()), (second.into(), "v1".into())]);

        // An empty utterance is answered too.
        tx.send(TtsInput::Flush).await.unwrap();
        assert!(matches!(next(&mut rx).await, TtsEvent::UtteranceDone));
    }

    #[tokio::test]
    async fn cancel_drops_the_inflight_synth_and_the_session_goes_on() {
        let fake = Arc::new(FakeTts::default());
        *fake.delay.lock().unwrap() = Duration::from_secs(30);
        let tts = BatchStreamingTts::new(Arc::clone(&fake), AudioFormat::f32(24_000));
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();

        tx.send(TtsInput::Text("A long utterance that takes forever.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        tx.send(TtsInput::Text("Queued behind it, never spoken.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        tx.send(TtsInput::Cancel).await.unwrap();
        assert!(matches!(next(&mut rx).await, TtsEvent::Cancelled));
        assert!(fake.dropped.load(Ordering::SeqCst), "the synthesize future was dropped");

        *fake.delay.lock().unwrap() = Duration::ZERO;
        tx.send(TtsInput::Text("After the cancel.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        assert!(matches!(next(&mut rx).await, TtsEvent::Audio(_)));
        assert!(matches!(next(&mut rx).await, TtsEvent::UtteranceDone));
        assert_eq!(fake.said.lock().unwrap().len(), 2, "the queued utterance was never synthesized");
    }

    #[tokio::test]
    async fn a_failed_utterance_is_an_error_and_the_next_one_plays() {
        let fake = Arc::new(FakeTts::default());
        let tts = BatchStreamingTts::new(Arc::clone(&fake), AudioFormat::pcm16(24_000));
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();

        tx.send(TtsInput::Text("Please fail this one right now. And skip this sentence.".into()))
            .await
            .unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        tx.send(TtsInput::Text("Fine.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let TtsEvent::Error(e) = next(&mut rx).await else { panic!("error") };
        assert!(e.contains("fake failure"));
        // PCM16 declared: the f32 samples come back as 2 bytes each.
        let TtsEvent::Audio(a) = next(&mut rx).await else { panic!("audio") };
        assert_eq!(a.data.len(), 2 * "Fine.".len());
        assert_eq!(i16::from_le_bytes([a.data[0], a.data[1]]), 16383);
        assert!(matches!(next(&mut rx).await, TtsEvent::UtteranceDone));
        assert_eq!(fake.said.lock().unwrap().len(), 2, "the failed utterance's rest was skipped");
    }

    #[tokio::test]
    async fn a_provider_at_another_rate_fails_the_utterance() {
        let fake = Arc::new(FakeTts::default());
        *fake.rate.lock().unwrap() = Some(16_000);
        let tts = BatchStreamingTts::new(Arc::clone(&fake), AudioFormat::f32(24_000));
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();
        tx.send(TtsInput::Text("Hello.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let TtsEvent::Error(e) = next(&mut rx).await else { panic!("error") };
        assert!(e.contains("16000 Hz"), "{e}");
    }

    #[tokio::test]
    async fn dropping_the_sender_speaks_what_is_queued_then_closes() {
        let fake = Arc::new(FakeTts::default());
        let tts = BatchStreamingTts::new(Arc::clone(&fake), AudioFormat::f32(24_000));
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();
        tx.send(TtsInput::Text("Unflushed words.".into())).await.unwrap();
        drop(tx);
        assert!(matches!(next(&mut rx).await, TtsEvent::Audio(_)));
        assert!(matches!(next(&mut rx).await, TtsEvent::UtteranceDone));
        assert!(rx.recv().await.is_none());
    }
}
