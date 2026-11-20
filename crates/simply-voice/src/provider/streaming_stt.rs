//! Streaming speech-to-text: audio frames in, transcript and turn events out.
//!
//! A [`StreamingStt`] session is a channel pair, meant to stay open for as
//! long as people may speak. The caller sends [`SttInput::Audio`] (PCM16 LE,
//! mono, at the rate it declared in [`StreamingSttConfig`]) and, when its own
//! voice activity detector hears the end of a turn, [`SttInput::EndOfTurn`].
//! It receives [`SttEvent`]s.
//!
//! **Turns.** Events are grouped into turns. A turn's text is its
//! [`SttEvent::Final`]s, appended in order; a [`SttEvent::Partial`] replaces
//! the previous partial of the same turn and is dropped by its next `Final`
//! or by the turn's end. **Every turn ends with exactly one
//! [`SttEvent::TurnEnded`]**, after all of its text, and nothing of that turn
//! comes after it. [`SttEvent::Error`] reports a failure without ending the
//! turn; the session goes on.
//!
//! **Who ends the turn** is in [`SttCaps::ends_turns`]:
//!
//! * A provider that **ends turns** (Mistral's turn detection, Kyutai's
//!   end-of-turn head) opens a turn with [`SttEvent::SpeechStarted`], may send
//!   [`SttEvent::EagerEnd`] / [`SttEvent::Resumed`] while it is open (when
//!   [`SttCaps::eager_end`]), and ends it on its own. It may ignore
//!   `EndOfTurn`. The caller keeps sending audio, silence included: the
//!   provider measures silence in the audio it receives.
//! * A provider that **doesn't** sends no `SpeechStarted`: the caller's
//!   detector owns the start of speech. Every `EndOfTurn` is answered, in
//!   order, by exactly one `TurnEnded` (with
//!   [`TurnEnd::Requested`]), its text before it; the turn's audio is
//!   everything sent since the previous `EndOfTurn`, so such a caller sends
//!   audio from its detector's start of speech (pre-roll included) to the end.
//!
//! **Closing.** Dropping the input sender finishes what is in flight (an open
//! turn ends with [`TurnEnd::Closed`]) and then closes the event receiver.
//! The caller must keep reading events: a full event channel holds the
//! session up.
//!
//! [`BatchStreamingStt`] puts any [`SttProvider`] behind this trait.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;

use super::stt::SttProvider;
use crate::audio::AudioChunk;

/// What a caller sends into a streaming STT session.
#[derive(Debug, Clone)]
pub enum SttInput {
    /// PCM16 LE mono audio at the session's declared sample rate.
    Audio(AudioChunk),
    /// The caller's voice activity detector heard the end of a turn. Providers
    /// that end turns themselves may ignore it (see the module docs).
    EndOfTurn,
}

/// Why a turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TurnEnd {
    /// The provider heard the speaker stop (Mistral: `end_reason: threshold`).
    Silence,
    /// The provider's longest pause ran out before it was sure the speaker
    /// had finished (Mistral: `end_reason: timeout`).
    Timeout,
    /// The caller sent [`SttInput::EndOfTurn`].
    Requested,
    /// The provider's connection dropped mid-turn. The turn's text is what
    /// arrived before; the session reconnects on its own.
    ConnectionLost,
    /// The input sender was dropped while the turn was open.
    Closed,
}

/// What a streaming STT session sends back. See the module docs for how
/// events group into turns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SttEvent {
    /// The provider heard speech start: a turn opens. Only from providers that
    /// end turns.
    SpeechStarted,
    /// Text that may still change: replaces the previous partial of this turn.
    Partial(String),
    /// Text appended to this turn's transcript, never revised. Carries its own
    /// spaces and punctuation: concatenate as is.
    Final(String),
    /// The provider thinks the speaker has probably finished; work started on
    /// it must be droppable. Only when [`SttCaps::eager_end`].
    EagerEnd,
    /// The speaker went on after an [`SttEvent::EagerEnd`].
    Resumed,
    /// The turn is over: its text is complete.
    TurnEnded { reason: TurnEnd },
    /// Something failed; the session goes on (a provider with a connection
    /// reconnects).
    Error(String),
}

/// What a streaming STT provider does, for the caller to plan around.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SttCaps {
    /// The provider detects speech and ends turns itself; the caller's
    /// detector is only a fallback (see the module docs).
    pub ends_turns: bool,
    /// The provider sends [`SttEvent::EagerEnd`] and [`SttEvent::Resumed`].
    pub eager_end: bool,
    /// The sample rate the provider prefers. [`StreamingSttConfig`] may
    /// declare another one where the provider accepts it.
    pub sample_rate: u32,
}

/// Settings for one streaming STT session.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamingSttConfig {
    /// The sample rate of the PCM16 audio the caller sends.
    pub sample_rate: u32,
}

impl StreamingSttConfig {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }
}

impl Default for StreamingSttConfig {
    /// The pipeline's rate, [`crate::audio::SAMPLE_RATE`] (16 kHz).
    fn default() -> Self {
        Self::new(crate::audio::SAMPLE_RATE)
    }
}

/// The sender and receiver of an open session.
pub type SttSession = (mpsc::Sender<SttInput>, mpsc::Receiver<SttEvent>);

/// Speech-to-text that takes audio as it is heard and returns text and turn
/// events as they happen. See the module documentation for the event contract.
#[async_trait]
pub trait StreamingStt: Send + Sync {
    fn caps(&self) -> SttCaps;

    /// Open a session. It lives until the input sender is dropped or the
    /// event receiver is.
    async fn open(&self, cfg: StreamingSttConfig) -> Result<SttSession>;
}

/// How many inputs and events a session buffers.
pub(crate) const SESSION_CHANNEL: usize = 256;

/// Fails unless `cfg` declares `rate`, for providers that take one rate only.
pub(crate) fn require_rate(cfg: &StreamingSttConfig, rate: u32, provider: &str) -> Result<()> {
    if cfg.sample_rate != rate {
        anyhow::bail!("{provider} takes {rate} Hz audio, the session declared {} Hz", cfg.sample_rate);
    }
    Ok(())
}

/// Puts a batch [`SttProvider`] behind [`StreamingStt`], with
/// `ends_turns: false`: audio is buffered, and each `EndOfTurn` transcribes
/// what arrived since the previous one and answers with `Final` (unless the
/// text is empty) and `TurnEnded`. A failed transcription is an `Error`
/// followed by the turn's `TurnEnded`.
///
/// **Hearing never stops**: transcriptions run one at a time, in order, beside
/// the intake, so audio that arrives during one is kept for the next turn.
pub struct BatchStreamingStt<P: ?Sized> {
    provider: Arc<P>,
    sample_rate: u32,
}

impl<P: SttProvider + ?Sized + 'static> BatchStreamingStt<P> {
    /// For a provider that takes [`crate::audio::SAMPLE_RATE`] (16 kHz), as
    /// every one in this crate does.
    pub fn new(provider: Arc<P>) -> Self {
        Self { provider, sample_rate: crate::audio::SAMPLE_RATE }
    }

    /// The one rate `provider` takes; sessions must declare it.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }
}

#[async_trait]
impl<P: SttProvider + ?Sized + 'static> StreamingStt for BatchStreamingStt<P> {
    fn caps(&self) -> SttCaps {
        SttCaps { ends_turns: false, eager_end: false, sample_rate: self.sample_rate }
    }

    async fn open(&self, cfg: StreamingSttConfig) -> Result<SttSession> {
        require_rate(&cfg, self.sample_rate, "this provider")?;
        let (in_tx, in_rx) = mpsc::channel(SESSION_CHANNEL);
        let (ev_tx, ev_rx) = mpsc::channel(SESSION_CHANNEL);
        tokio::spawn(run_batch(Arc::clone(&self.provider), in_rx, ev_tx));
        Ok((in_tx, ev_rx))
    }
}

type Transcribe = Pin<Box<dyn Future<Output = Result<String>> + Send>>;

async fn run_batch<P: SttProvider + ?Sized + 'static>(
    provider: Arc<P>,
    mut input: mpsc::Receiver<SttInput>,
    events: mpsc::Sender<SttEvent>,
) {
    // Audio since the last EndOfTurn.
    let mut audio: Vec<u8> = Vec::new();
    // Ended turns waiting for a transcription, oldest first.
    let mut queue: VecDeque<(Vec<u8>, TurnEnd)> = VecDeque::new();
    let mut inflight: Option<(Transcribe, TurnEnd)> = None;
    let mut input_open = true;

    loop {
        // Start the next transcription; turns with no audio end at once.
        while inflight.is_none() {
            let Some((pcm, reason)) = queue.pop_front() else { break };
            if pcm.is_empty() {
                if events.send(SttEvent::TurnEnded { reason }).await.is_err() {
                    return;
                }
                continue;
            }
            let provider = Arc::clone(&provider);
            let fut: Transcribe =
                Box::pin(async move { provider.transcribe(AudioChunk::new(pcm)).await.map(|t| t.text) });
            inflight = Some((fut, reason));
        }
        if !input_open && inflight.is_none() {
            return;
        }

        tokio::select! {
            biased;
            // The transcription is polled first: a steady stream of audio
            // must not starve it.
            res = async { inflight.as_mut().expect("guarded").0.as_mut().await }, if inflight.is_some() => {
                let (_, reason) = inflight.take().expect("guarded");
                let first = match res {
                    Ok(text) if text.trim().is_empty() => None,
                    Ok(text) => Some(SttEvent::Final(text.trim().to_string())),
                    Err(e) => Some(SttEvent::Error(format!("{e:#}"))),
                };
                for ev in first.into_iter().chain([SttEvent::TurnEnded { reason }]) {
                    if events.send(ev).await.is_err() {
                        return;
                    }
                }
            }
            inp = input.recv(), if input_open => match inp {
                Some(SttInput::Audio(chunk)) => audio.extend_from_slice(&chunk.data),
                Some(SttInput::EndOfTurn) => queue.push_back((std::mem::take(&mut audio), TurnEnd::Requested)),
                None => {
                    input_open = false;
                    if !audio.is_empty() {
                        queue.push_back((std::mem::take(&mut audio), TurnEnd::Closed));
                    }
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Transcription;
    use std::sync::Mutex;
    use std::time::Duration;

    /// Answers each call, after `delay`, with the number of bytes it got;
    /// records every call's length; fails audio starting with byte 0xFF.
    #[derive(Default)]
    struct FakeStt {
        calls: Mutex<Vec<usize>>,
        delay: Mutex<Duration>,
    }

    #[async_trait]
    impl SttProvider for FakeStt {
        async fn transcribe(&self, audio: AudioChunk) -> Result<Transcription> {
            self.calls.lock().unwrap().push(audio.data.len());
            let delay = *self.delay.lock().unwrap();
            tokio::time::sleep(delay).await;
            if audio.data.first() == Some(&0xFF) {
                anyhow::bail!("fake failure");
            }
            Ok(Transcription { text: format!(" {} bytes ", audio.data.len()), is_final: true })
        }
        async fn stream(&self) -> Result<(mpsc::Sender<AudioChunk>, mpsc::Receiver<Transcription>)> {
            unimplemented!()
        }
    }

    async fn next(rx: &mut mpsc::Receiver<SttEvent>) -> SttEvent {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event in time")
            .expect("session open")
    }

    fn chunk(n: usize) -> SttInput {
        SttInput::Audio(AudioChunk::new(vec![1u8; n]))
    }

    #[tokio::test]
    async fn each_end_of_turn_transcribes_the_audio_since_the_last_one() {
        let fake = Arc::new(FakeStt::default());
        let stt = BatchStreamingStt::new(Arc::clone(&fake));
        assert!(!stt.caps().ends_turns);
        let (tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();

        tx.send(chunk(640)).await.unwrap();
        tx.send(chunk(640)).await.unwrap();
        tx.send(SttInput::EndOfTurn).await.unwrap();
        assert_eq!(next(&mut rx).await, SttEvent::Final("1280 bytes".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Requested });

        // A turn with no audio is answered without a call.
        tx.send(SttInput::EndOfTurn).await.unwrap();
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Requested });
        assert_eq!(*fake.calls.lock().unwrap(), vec![1280]);
    }

    #[tokio::test]
    async fn audio_during_a_slow_transcription_is_kept_for_the_next_turn() {
        let fake = Arc::new(FakeStt::default());
        *fake.delay.lock().unwrap() = Duration::from_millis(300);
        let stt = BatchStreamingStt::new(Arc::clone(&fake));
        let (tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();

        tx.send(chunk(640)).await.unwrap();
        tx.send(SttInput::EndOfTurn).await.unwrap();
        // While the first call sleeps: more chunks than the input channel
        // holds, each sent at once. A blocked intake would stall here.
        let fed = tokio::time::timeout(Duration::from_millis(250), async {
            for _ in 0..(SESSION_CHANNEL * 2) {
                tx.send(chunk(640)).await.unwrap();
            }
        })
        .await;
        assert!(fed.is_ok(), "intake kept up during the transcription");
        assert_eq!(fake.calls.lock().unwrap().len(), 1, "the first call is still running");
        tx.send(SttInput::EndOfTurn).await.unwrap();

        assert_eq!(next(&mut rx).await, SttEvent::Final("640 bytes".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Requested });
        let second = 640 * SESSION_CHANNEL * 2;
        assert_eq!(next(&mut rx).await, SttEvent::Final(format!("{second} bytes")));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Requested });
        assert_eq!(*fake.calls.lock().unwrap(), vec![640, second], "nothing dropped");
    }

    #[tokio::test]
    async fn a_failed_transcription_is_an_error_and_still_ends_its_turn() {
        let fake = Arc::new(FakeStt::default());
        let stt = BatchStreamingStt::new(Arc::clone(&fake));
        let (tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        tx.send(SttInput::Audio(AudioChunk::new(vec![0xFFu8; 4]))).await.unwrap();
        tx.send(SttInput::EndOfTurn).await.unwrap();
        tx.send(chunk(2)).await.unwrap();
        tx.send(SttInput::EndOfTurn).await.unwrap();
        let SttEvent::Error(e) = next(&mut rx).await else { panic!("error") };
        assert!(e.contains("fake failure"));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Requested });
        assert_eq!(next(&mut rx).await, SttEvent::Final("2 bytes".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Requested });
    }

    #[tokio::test]
    async fn dropping_the_sender_transcribes_the_rest_then_closes() {
        let fake = Arc::new(FakeStt::default());
        let stt = BatchStreamingStt::new(Arc::clone(&fake));
        let (tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        tx.send(chunk(10)).await.unwrap();
        drop(tx);
        assert_eq!(next(&mut rx).await, SttEvent::Final("10 bytes".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Closed });
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn open_refuses_another_sample_rate() {
        let stt = BatchStreamingStt::new(Arc::new(FakeStt::default()));
        assert!(stt.open(StreamingSttConfig::new(24_000)).await.is_err());
        let stt = BatchStreamingStt::new(Arc::new(FakeStt::default())).with_sample_rate(24_000);
        assert!(stt.open(StreamingSttConfig::new(24_000)).await.is_ok());
    }
}
