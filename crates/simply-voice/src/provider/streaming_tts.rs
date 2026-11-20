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
//! [`BatchStreamingTts`] puts any [`TtsProvider`] behind this trait, and
//! [`SentenceStreamingTts`] any provider that speaks one text per request
//! with streamed audio ([`SentenceTts`]); both share one worker and the
//! sentence rule of [`SentenceChunker`].

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{self, BoxStream, StreamExt};
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

/// One request's audio as it is generated, in the format its provider
/// declares. An `Err` ends the request.
pub type AudioStream = BoxStream<'static, Result<Bytes>>;

/// A provider that speaks one text per request and streams the audio back
/// (Gemini TTS, for one). [`SentenceStreamingTts`] puts it behind
/// [`StreamingTts`].
pub trait SentenceTts: Send + Sync {
    /// The format of every chunk [`SentenceTts::speak`] yields.
    fn format(&self) -> AudioFormat;

    /// Speak `text` in `voice` (empty: the provider's default). The request
    /// starts when the stream is first polled; dropping the stream drops it.
    fn speak(&self, text: String, voice: String) -> AudioStream;
}

/// Puts a per-request [`SentenceTts`] behind [`StreamingTts`]: text deltas
/// are cut into sentences as they arrive ([`SentenceChunker`]), and each
/// sentence is one request, spoken in order, its audio sent on as it streams.
/// The first sentence starts before the `Flush`; the `Flush` sends the
/// trailing fragment. `Cancel` drops the request in flight and the queued
/// sentences. Prosody starts afresh at each sentence.
pub struct SentenceStreamingTts<P: ?Sized> {
    provider: Arc<P>,
}

impl<P: SentenceTts + ?Sized + 'static> SentenceStreamingTts<P> {
    pub fn new(provider: Arc<P>) -> Self {
        Self { provider }
    }
}

#[async_trait]
impl<P: SentenceTts + ?Sized + 'static> StreamingTts for SentenceStreamingTts<P> {
    async fn open(&self, cfg: StreamingTtsConfig) -> Result<TtsSession> {
        let format = self.provider.format();
        Ok(spawn_sentences(Arc::clone(&self.provider), cfg.voice, true, SentenceChunker::new(), format))
    }
}

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
        let synth = Arc::new(BatchSentences { provider: Arc::clone(&self.provider), format: self.format });
        let chunker = if self.split_sentences { SentenceChunker::new() } else { SentenceChunker::whole() };
        Ok(spawn_sentences(synth, cfg.voice, false, chunker, self.format))
    }
}

/// A batch provider as a [`SentenceTts`]: each request is one `synthesize`,
/// converted to `format`.
struct BatchSentences<P: ?Sized> {
    provider: Arc<P>,
    format: AudioFormat,
}

impl<P: TtsProvider + ?Sized + 'static> SentenceTts for BatchSentences<P> {
    fn format(&self) -> AudioFormat {
        self.format
    }

    fn speak(&self, text: String, voice: String) -> AudioStream {
        let provider = Arc::clone(&self.provider);
        let format = self.format;
        stream::once(async move { provider.synthesize(&text, &voice).await.and_then(|a| into_format(a, format)) })
            .boxed()
    }
}

/// Open a session over `synth`. `eager` starts an utterance's sentences as
/// they complete; otherwise they wait for its `Flush`.
fn spawn_sentences<S: SentenceTts + ?Sized + 'static>(
    synth: Arc<S>,
    voice: String,
    eager: bool,
    chunker: SentenceChunker,
    format: AudioFormat,
) -> TtsSession {
    let (in_tx, in_rx) = mpsc::channel(SESSION_CHANNEL);
    let (ev_tx, ev_rx) = mpsc::channel(SESSION_CHANNEL);
    let worker = SentenceWorker { synth, voice, eager, events: ev_tx };
    tokio::spawn(worker.run(in_rx, chunker));
    (format, in_tx, ev_rx)
}

/// One utterance's sentences not yet started.
#[derive(Default)]
struct Utterance {
    sentences: VecDeque<String>,
    /// Its `Flush` arrived: no more text comes.
    flushed: bool,
    /// A sentence of it was started.
    started: bool,
    /// A sentence of it failed: the rest is skipped and its `Flush` is
    /// answered with this.
    failed: Option<String>,
}

struct SentenceWorker<S: ?Sized> {
    synth: Arc<S>,
    voice: String,
    eager: bool,
    events: mpsc::Sender<TtsEvent>,
}

/// The utterance text goes into: the last one, unless it is flushed.
fn open_utterance(queue: &mut VecDeque<Utterance>) -> &mut Utterance {
    if queue.back().is_none_or(|u| u.flushed) {
        queue.push_back(Utterance::default());
    }
    queue.back_mut().expect("just pushed")
}

impl<S: SentenceTts + ?Sized + 'static> SentenceWorker<S> {
    async fn run(self, mut input: mpsc::Receiver<TtsInput>, mut chunker: SentenceChunker) {
        // Utterances in order: the front one is spoken, the back one may still
        // be taking text.
        let mut queue: VecDeque<Utterance> = VecDeque::new();
        let mut inflight: Option<AudioStream> = None;
        let mut input_open = true;

        loop {
            while inflight.is_none() {
                let Some(front) = queue.front_mut() else { break };
                if front.failed.is_some() {
                    front.sentences.clear();
                }
                if front.flushed && front.sentences.is_empty() {
                    let ev = match queue.pop_front().and_then(|u| u.failed) {
                        Some(e) => TtsEvent::Error(e),
                        None => TtsEvent::UtteranceDone,
                    };
                    if self.events.send(ev).await.is_err() {
                        return;
                    }
                    continue;
                }
                if !(self.eager || front.flushed) {
                    break;
                }
                match front.sentences.pop_front() {
                    Some(s) => {
                        front.started = true;
                        inflight = Some(self.synth.speak(s, self.voice.clone()));
                    }
                    None => break,
                }
            }
            if !input_open && inflight.is_none() && queue.is_empty() {
                return;
            }

            tokio::select! {
                biased;
                inp = input.recv(), if input_open => match inp {
                    Some(TtsInput::Text(t)) => {
                        let done = chunker.push(&t);
                        open_utterance(&mut queue).sentences.extend(done);
                    }
                    Some(TtsInput::Flush) => {
                        let rest = chunker.finish();
                        let u = open_utterance(&mut queue);
                        u.sentences.extend(rest);
                        u.flushed = true;
                    }
                    Some(TtsInput::Cancel) => {
                        chunker.clear();
                        queue.clear();
                        inflight = None; // drops the request
                        if self.events.send(TtsEvent::Cancelled).await.is_err() {
                            return;
                        }
                    }
                    None => {
                        input_open = false;
                        let rest = chunker.finish();
                        if let Some(u) = queue.back_mut().filter(|u| !u.flushed) {
                            u.sentences.extend(rest);
                            if u.sentences.is_empty() && !u.started && u.failed.is_none() {
                                queue.pop_back(); // whitespace only: nothing to say
                            } else {
                                u.flushed = true;
                            }
                        }
                    }
                },
                chunk = async { inflight.as_mut().expect("guarded").next().await }, if inflight.is_some() => match chunk {
                    Some(Ok(data)) => {
                        if !data.is_empty() && self.events.send(TtsEvent::Audio(AudioChunk::new(data))).await.is_err() {
                            return;
                        }
                    }
                    Some(Err(e)) => {
                        inflight = None;
                        if let Some(front) = queue.front_mut() {
                            front.failed.get_or_insert(format!("{e:#}"));
                        }
                    }
                    None => inflight = None,
                },
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

/// Cuts text that arrives in pieces into sentences as each one completes: after
/// a sentence end (`.`, `!`, `?`, `…`) once the whitespace after it arrives,
/// and after each line break; pieces shorter than 24 characters are joined to
/// the next; each sentence is trimmed. [`split_sentences`] is the same rule
/// over a whole text.
#[derive(Debug, Clone, Default)]
pub struct SentenceChunker {
    piece: String,
    /// The last character was a sentence end: the next decides.
    after_end: bool,
    /// Never cut: [`SentenceChunker::finish`] returns everything.
    whole: bool,
}

impl SentenceChunker {
    pub fn new() -> Self {
        Self::default()
    }

    /// A chunker that never cuts.
    pub(crate) fn whole() -> Self {
        Self { whole: true, ..Self::default() }
    }

    /// Take a piece of text; returns the sentences it completed.
    pub fn push(&mut self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        if self.whole {
            self.piece.push_str(text);
            return out;
        }
        for c in text.chars() {
            if std::mem::take(&mut self.after_end) && c.is_whitespace() {
                self.cut(&mut out);
            }
            self.piece.push(c);
            if c == '\n' {
                self.cut(&mut out);
            } else if matches!(c, '.' | '!' | '?' | '…') {
                self.after_end = true;
            }
        }
        out
    }

    /// The text is complete: the rest, trimmed, unless blank.
    pub fn finish(&mut self) -> Option<String> {
        self.after_end = false;
        let rest = std::mem::take(&mut self.piece);
        let rest = rest.trim();
        (!rest.is_empty()).then(|| rest.to_string())
    }

    /// Drop the text taken so far.
    pub fn clear(&mut self) {
        self.piece.clear();
        self.after_end = false;
    }

    fn cut(&mut self, out: &mut Vec<String>) {
        let s = self.piece.trim();
        if s.chars().count() >= MIN_SENTENCE_CHARS {
            out.push(s.to_string());
            self.piece.clear();
        }
    }
}

/// `text` cut after each sentence end (`.`, `!`, `?`, `…` followed by
/// whitespace) and each line break, short pieces joined to the next, each
/// trimmed; nothing for blank text.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut chunker = SentenceChunker::new();
    let mut out = chunker.push(text);
    out.extend(chunker.finish());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioEncoding;
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

    #[test]
    fn the_chunker_cuts_as_text_arrives_by_the_same_rule() {
        let text = "Hi. This is the first real sentence. And a second one, also long!\nok. Version 1.5 is out";
        // Any split of the input gives what split_sentences gives at once.
        for at in [1, 4, 17, 36, 37, 60, text.len() - 1] {
            let mut c = SentenceChunker::new();
            let mut got = c.push(&text[..at]);
            got.extend(c.push(&text[at..]));
            got.extend(c.finish());
            assert_eq!(got, split_sentences(text), "split at {at}");
        }
        // A sentence end waits for the whitespace after it.
        let mut c = SentenceChunker::new();
        assert!(c.push("This sentence is long enough.").is_empty());
        assert_eq!(c.push(" And"), vec!["This sentence is long enough."]);
        assert_eq!(c.finish(), Some("And".to_string()));
        assert_eq!(c.finish(), None);
    }

    /// Speaks each text as two chunks of one byte per character (as PCM16
    /// pairs), recording the texts asked; a text containing "stall" sends
    /// its first chunk and then hangs, setting `dropped` when dropped; one
    /// containing "fail" fails.
    #[derive(Default)]
    struct FakeSentences {
        said: Mutex<Vec<String>>,
        dropped: Arc<AtomicBool>,
    }

    impl SentenceTts for FakeSentences {
        fn format(&self) -> AudioFormat {
            AudioFormat::pcm16(24_000)
        }
        fn speak(&self, text: String, _voice: String) -> AudioStream {
            self.said.lock().unwrap().push(text.clone());
            let dropped = Arc::clone(&self.dropped);
            let half = Bytes::from(vec![1u8; text.len()]);
            stream::unfold(0, move |step| {
                let (text, half, dropped) = (text.clone(), half.clone(), Arc::clone(&dropped));
                async move {
                    match step {
                        0 if text.contains("fail") => Some((Err(anyhow::anyhow!("fake failure")), 9)),
                        0 => Some((Ok(half), 1)),
                        1 if text.contains("stall") => {
                            let _flag = DropFlag(dropped, false);
                            std::future::pending::<()>().await;
                            None
                        }
                        1 => Some((Ok(half), 2)),
                        _ => None,
                    }
                }
            })
            .boxed()
        }
    }

    #[tokio::test]
    async fn sentences_start_before_the_flush_and_play_in_order() {
        let fake = Arc::new(FakeSentences::default());
        let tts = SentenceStreamingTts::new(Arc::clone(&fake));
        let (format, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();
        assert_eq!((format.encoding, format.sample_rate), (AudioEncoding::Pcm16, 24_000));

        let first = "The first sentence is long enough.";
        let second = "And so is the second sentence here.";
        tx.send(TtsInput::Text(format!("{first} {}", &second[..10]))).await.unwrap();
        for _ in 0..2 {
            let TtsEvent::Audio(a) = next(&mut rx).await else { panic!("audio") };
            assert_eq!(a.data.len(), first.len());
        }
        assert_eq!(fake.said.lock().unwrap().clone(), vec![first.to_string()], "spoken before the flush");

        tx.send(TtsInput::Text(format!("{} Trailing bit", &second[10..]))).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        for len in [second.len(), second.len(), 12, 12] {
            let TtsEvent::Audio(a) = next(&mut rx).await else { panic!("audio") };
            assert_eq!(a.data.len(), len);
        }
        assert!(matches!(next(&mut rx).await, TtsEvent::UtteranceDone));
        assert_eq!(fake.said.lock().unwrap().clone(), vec![first, second, "Trailing bit"]);
    }

    #[tokio::test]
    async fn cancel_mid_sentence_drops_the_request_and_the_queue() {
        let fake = Arc::new(FakeSentences::default());
        let tts = SentenceStreamingTts::new(Arc::clone(&fake));
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();
        tx.send(TtsInput::Text("This one will stall midway through. Queued and never said. ".into()))
            .await
            .unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        assert!(matches!(next(&mut rx).await, TtsEvent::Audio(_)));
        tx.send(TtsInput::Cancel).await.unwrap();
        assert!(matches!(next(&mut rx).await, TtsEvent::Cancelled));
        assert!(fake.dropped.load(Ordering::SeqCst), "the stalled request was dropped");

        tx.send(TtsInput::Text("After the cancel.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        assert!(matches!(next(&mut rx).await, TtsEvent::Audio(_)));
        assert!(matches!(next(&mut rx).await, TtsEvent::Audio(_)));
        assert!(matches!(next(&mut rx).await, TtsEvent::UtteranceDone));
        assert_eq!(fake.said.lock().unwrap().len(), 2, "the queued sentence was never requested");
    }

    #[tokio::test]
    async fn a_sentence_failing_before_the_flush_is_answered_at_the_flush() {
        let fake = Arc::new(FakeSentences::default());
        let tts = SentenceStreamingTts::new(Arc::clone(&fake));
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();
        tx.send(TtsInput::Text("Please fail on this sentence. ".into())).await.unwrap();
        tx.send(TtsInput::Text("Then this is skipped. ".into())).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(rx.try_recv().is_err(), "nothing answers before the flush");
        tx.send(TtsInput::Flush).await.unwrap();
        let TtsEvent::Error(e) = next(&mut rx).await else { panic!("error") };
        assert!(e.contains("fake failure"), "{e}");
        tx.send(TtsInput::Text("Fine.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        assert!(matches!(next(&mut rx).await, TtsEvent::Audio(_)));
        assert!(matches!(next(&mut rx).await, TtsEvent::Audio(_)));
        assert!(matches!(next(&mut rx).await, TtsEvent::UtteranceDone));
        assert_eq!(fake.said.lock().unwrap().clone(), vec!["Please fail on this sentence.", "Fine."]);
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
