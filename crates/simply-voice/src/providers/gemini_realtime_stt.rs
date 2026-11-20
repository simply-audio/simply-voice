//! Gemini live transcription: audio in over the Live API's WebSocket,
//! interim and final text and the server's turn signals out, with hybrid
//! voice activity detection.
//!
//! Protocol (Google's *Live transcription*, *Session management* and the
//! *Live API WebSockets reference*): connect to
//! `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=…`,
//! send `{setup: {model: "models/gemini-3.5-transcribe-live", generationConfig:
//! {responseModalities: ["TEXT"]}, inputAudioTranscription: {…}}}` and wait
//! for `setupComplete`; then `{realtimeInput: {audio: {data, mimeType:
//! "audio/pcm;rate=16000"}}}` (about 100 ms per message), and
//! `{realtimeInput: {audioStreamEnd: true}}` when the client's detector hears
//! the end of speech. The server answers with `serverContent`
//! (`interimInputTranscription`, `inputTranscription`, `turnComplete`,
//! `generationComplete`), `voiceActivity` (`ACTIVITY_START` / `ACTIVITY_END`),
//! and `goAway` before it drops the connection (about 10 minutes in). The
//! server may send its JSON in binary frames; both kinds are read.
//!
//! **Mapping.** `voiceActivity` `ACTIVITY_START` (or the first transcription
//! of a turn, if it comes first) → [`SttEvent::SpeechStarted`];
//! `interimInputTranscription` → [`SttEvent::Partial`];
//! `inputTranscription` → [`SttEvent::Final`] (`VERBATIM` mode: one final per
//! segment, appended); the first of `ACTIVITY_END`, `generationComplete` and
//! `turnComplete` ends the turn ([`TurnEnd::Silence`], or
//! [`TurnEnd::Requested`] after an `EndOfTurn`). Transcriptions are not
//! ordered against the other messages, so a turn whose last interim has no
//! final yet stays open for its final (at most
//! [`GeminiRealtimeStt::with_end_hold`]); if none comes, the last interim
//! becomes its `Final`. A final with no turn open is a turn of its own, with
//! no `SpeechStarted`.
//!
//! **Hybrid VAD.** The server detects speech starts and, as a fallback, ends.
//! [`SttInput::EndOfTurn`] sends `audioStreamEnd` and the server finalizes at
//! once; the turn ends with `Requested`. If the server answers nothing within
//! [`GeminiRealtimeStt::with_end_timeout`], the turn ends anyway.
//!
//! **Connections.** A connection lives about 10 minutes. On `goAway`, or
//! [`GeminiRealtimeStt::with_rotate_after`] into a connection, a new one is
//! opened and swapped in at the next turn boundary. A connection that drops
//! is replaced with backoff; audio sent meanwhile is kept, and an open turn
//! stays open: the audio sent since its last final is sent again on the new
//! connection, so nothing said in it is lost. Only a failed reconnect ends it,
//! with [`TurnEnd::ConnectionLost`] and one [`SttEvent::Error`] per outage.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use bytes::{Bytes, BytesMut};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, info, warn};

use super::realtime_ws::{self, Ws};
use crate::provider::streaming_stt::SESSION_CHANNEL;
use crate::provider::{SttCaps, SttEvent, SttInput, SttSession, StreamingStt, StreamingSttConfig, TurnEnd};

/// The live transcription model.
pub const DEFAULT_REALTIME_STT_MODEL: &str = "gemini-3.5-transcribe-live";
const DEFAULT_WS_BASE: &str = "https://generativelanguage.googleapis.com";
const WS_PATH: &str = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

/// Names this endpoint in errors.
const LABEL: &str = "gemini live STT";

/// How long the last words may take after the input sender is dropped.
const FINISH_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_BACKOFF: Duration = Duration::from_secs(5);
/// A connection lost sooner than this after it opened counts as a failed
/// attempt.
const STABLE_AFTER: Duration = Duration::from_secs(2);
/// Audio kept for a reconnect while no turn is open: enough for a start of
/// speech the server has not flagged yet.
const IDLE_REPLAY: Duration = Duration::from_secs(1);

/// Gemini's live transcription behind [`StreamingStt`], with hybrid VAD.
/// Takes PCM16 at any sample rate the session declares (the server resamples);
/// 16 kHz is native.
#[derive(Debug, Clone)]
pub struct GeminiRealtimeStt {
    api_key: String,
    base_url: String,
    model: String,
    language_codes: Vec<String>,
    custom_vocabulary: Vec<String>,
    chunk: Duration,
    end_hold: Duration,
    end_timeout: Duration,
    rotate_after: Duration,
    handshake_timeout: Duration,
    backlog: Duration,
    replay: Duration,
}

impl GeminiRealtimeStt {
    /// Google's hosted API, model `gemini-3.5-transcribe-live`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_WS_BASE.to_string(),
            model: DEFAULT_REALTIME_STT_MODEL.to_string(),
            language_codes: Vec::new(),
            custom_vocabulary: Vec::new(),
            chunk: Duration::from_millis(100),
            end_hold: Duration::from_millis(600),
            end_timeout: Duration::from_secs(2),
            rotate_after: Duration::from_secs(9 * 60),
            handshake_timeout: Duration::from_secs(10),
            backlog: Duration::from_secs(10),
            replay: Duration::from_secs(30),
        }
    }

    /// The server's root (`https://…` → `wss://…`, `http://…` → `ws://…`).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// BCP-47 language hints (`["en-US"]`); none detects the language.
    pub fn with_language_codes(mut self, codes: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.language_codes = codes.into_iter().map(Into::into).collect();
        self
    }

    /// Terms to bias recognition toward (names, jargon); up to 1,000, best
    /// under 100.
    pub fn with_custom_vocabulary(mut self, terms: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.custom_vocabulary = terms.into_iter().map(Into::into).collect();
        self
    }

    /// How much audio goes in one message; 100 ms by default, as Google
    /// recommends. An `EndOfTurn` sends what is gathered at once.
    pub fn with_chunk(mut self, chunk: Duration) -> Self {
        self.chunk = chunk;
        self
    }

    /// How long a turn the server ended stays open for a final still
    /// missing; 600 ms by default.
    pub fn with_end_hold(mut self, hold: Duration) -> Self {
        self.end_hold = hold;
        self
    }

    /// How long after an `EndOfTurn` the turn ends even if the server says
    /// nothing; 2 s by default.
    pub fn with_end_timeout(mut self, timeout: Duration) -> Self {
        self.end_timeout = timeout;
        self
    }

    /// How old a connection may get before it is replaced at the next turn
    /// boundary; 9 minutes by default (the server's limit is about 10).
    pub fn with_rotate_after(mut self, after: Duration) -> Self {
        self.rotate_after = after;
        self
    }

    /// How long connecting and the setup may take; 10 s by default.
    pub fn with_handshake_timeout(mut self, timeout: Duration) -> Self {
        self.handshake_timeout = timeout;
        self
    }

    /// How much audio is kept while there is no connection; 10 s by default.
    pub fn with_backlog(mut self, backlog: Duration) -> Self {
        self.backlog = backlog;
        self
    }

    /// How much of an open turn's audio is sent again after a dropped
    /// connection; 30 s by default.
    pub fn with_replay(mut self, replay: Duration) -> Self {
        self.replay = replay;
        self
    }

    fn url(&self) -> String {
        realtime_ws::ws_url(&self.base_url, &format!("{WS_PATH}?key={}", self.api_key))
    }

    fn setup(&self) -> Value {
        let mut transcription = json!({});
        if !self.language_codes.is_empty() {
            transcription["languageCodes"] = json!(self.language_codes);
        }
        if !self.custom_vocabulary.is_empty() {
            transcription["customVocabulary"] = json!(self.custom_vocabulary);
        }
        json!({"setup": {
            "model": format!("models/{}", self.model),
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": transcription,
        }})
    }
}

#[async_trait]
impl StreamingStt for GeminiRealtimeStt {
    fn caps(&self) -> SttCaps {
        SttCaps { ends_turns: true, eager_end: false, sample_rate: crate::audio::SAMPLE_RATE }
    }

    async fn open(&self, cfg: StreamingSttConfig) -> Result<SttSession> {
        let dial = Arc::new(Dial {
            url: self.url(),
            setup: self.setup().to_string(),
            timeout: self.handshake_timeout,
        });
        info!(model = %self.model, "gemini live STT: opening session");
        let first = dial.connect().await?;

        let bytes_per = |d: Duration| (d.as_secs_f64() * cfg.sample_rate as f64 * 2.0) as usize;
        let (in_tx, in_rx) = mpsc::channel(SESSION_CHANNEL);
        let (ev_tx, ev_rx) = mpsc::channel(SESSION_CHANNEL);
        let worker = Worker {
            dial,
            mime: format!("audio/pcm;rate={}", cfg.sample_rate),
            active: Some(first),
            connected_at: Instant::now(),
            connecting: None,
            standby: None,
            rotate_due: false,
            rotate_after: self.rotate_after,
            failures: 0,
            outage_reported: false,
            gather: BytesMut::new(),
            gather_since: None,
            chunk: self.chunk,
            chunk_bytes: bytes_per(self.chunk).max(2),
            backlog: Buffer::new(bytes_per(self.backlog)),
            replay: Buffer::new(bytes_per(self.replay)),
            idle_replay: bytes_per(IDLE_REPLAY),
            turn: Turn::Idle,
            interim: None,
            end_requested: false,
            end_deadline: None,
            end_hold: self.end_hold,
            end_timeout: self.end_timeout,
            events: ev_tx,
            gone: false,
            finishing: false,
        };
        tokio::spawn(worker.run(in_rx));
        Ok((in_tx, ev_rx))
    }
}

/// How to open and set up one connection.
struct Dial {
    /// Carries the key: never logged.
    url: String,
    setup: String,
    timeout: Duration,
}

impl Dial {
    async fn connect(&self) -> Result<Ws> {
        tokio::time::timeout(self.timeout, self.handshake())
            .await
            .map_err(|_| anyhow::anyhow!("{LABEL}: handshake timed out"))?
    }

    async fn handshake(&self) -> Result<Ws> {
        let mut ws = realtime_ws::connect(&self.url, "").await.map_err(|e| {
            // tungstenite errors may quote the URL, and with it the key.
            anyhow::anyhow!("{LABEL}: connect failed: {}", redact(&e.to_string()))
        })?;
        ws.send(Message::Text(self.setup.clone().into())).await.context("gemini live STT: sending setup")?;
        loop {
            match ws.next().await {
                Some(Ok(Message::Close(frame))) => {
                    let why = frame.map(|f| f.reason.to_string()).unwrap_or_default();
                    anyhow::bail!("{LABEL}: the server closed during setup: {why}");
                }
                Some(Ok(msg)) => {
                    if let Some(v) = json_of(msg) {
                        if v.get("setupComplete").is_some() {
                            return Ok(ws);
                        }
                    }
                }
                Some(Err(e)) => anyhow::bail!("{LABEL}: waiting for setupComplete: {}", redact(&e.to_string())),
                None => anyhow::bail!("{LABEL}: closed before setupComplete"),
            }
        }
    }
}

/// `text` with anything after `key=` hidden.
fn redact(text: &str) -> String {
    match text.find("key=") {
        Some(i) => {
            let rest = &text[i + 4..];
            let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_')).unwrap_or(rest.len());
            format!("{}key=…{}", &text[..i], &rest[end..])
        }
        None => text.to_string(),
    }
}

/// The JSON in a text or binary frame.
fn json_of(msg: Message) -> Option<Value> {
    match msg {
        Message::Text(t) => serde_json::from_str(&t).ok(),
        Message::Binary(b) => serde_json::from_slice(&b).ok(),
        _ => None,
    }
}

/// Where the current turn is.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Turn {
    Idle,
    Open,
    /// The server ended it (or it is a stray final); it waits for a final
    /// until `until`, then ends with `reason`.
    Holding { until: Instant, reason: TurnEnd },
}

/// Audio chunks, oldest first, the oldest dropped past `cap` bytes.
struct Buffer {
    chunks: VecDeque<Bytes>,
    bytes: usize,
    cap: usize,
}

impl Buffer {
    fn new(cap: usize) -> Self {
        Self { chunks: VecDeque::new(), bytes: 0, cap }
    }

    fn push(&mut self, pcm: Bytes) {
        self.bytes += pcm.len();
        self.chunks.push_back(pcm);
        self.trim(self.cap);
    }

    fn trim(&mut self, cap: usize) {
        while self.bytes > cap {
            match self.chunks.pop_front() {
                Some(old) => self.bytes -= old.len(),
                None => break,
            }
        }
    }

    fn clear(&mut self) {
        self.chunks.clear();
        self.bytes = 0;
    }

    fn take(&mut self) -> VecDeque<Bytes> {
        self.bytes = 0;
        std::mem::take(&mut self.chunks)
    }
}

type Connecting = Pin<Box<dyn Future<Output = Result<Ws>> + Send>>;

async fn read(active: &mut Option<Ws>) -> Option<Result<Message, tokio_tungstenite::tungstenite::Error>> {
    match active {
        Some(ws) => ws.next().await,
        None => std::future::pending().await,
    }
}

async fn sleep_until(at: Option<Instant>) {
    match at {
        Some(at) => tokio::time::sleep_until(at).await,
        None => std::future::pending().await,
    }
}

fn audio_message(pcm: &[u8], mime: &str) -> Message {
    Message::Text(json!({"realtimeInput": {"audio": {"data": STANDARD.encode(pcm), "mimeType": mime}}}).to_string().into())
}

fn stream_end() -> Message {
    Message::Text(json!({"realtimeInput": {"audioStreamEnd": true}}).to_string().into())
}

/// One session: owns the connections and assembles turns.
struct Worker {
    dial: Arc<Dial>,
    mime: String,
    active: Option<Ws>,
    connected_at: Instant,
    /// A new connection being opened (a reconnect, or a rotation).
    connecting: Option<Connecting>,
    /// A rotation's new connection, waiting for the open turn to end.
    standby: Option<Ws>,
    /// `active` should be replaced at the next turn boundary.
    rotate_due: bool,
    rotate_after: Duration,
    failures: u32,
    /// This outage's failed reconnect was reported.
    outage_reported: bool,
    /// Audio gathered toward the next message.
    gather: BytesMut,
    gather_since: Option<Instant>,
    chunk: Duration,
    chunk_bytes: usize,
    /// Audio received while there was no connection.
    backlog: Buffer,
    /// Audio sent on `active` since the turn's last final (while idle, only
    /// the last second): sent again if the connection drops.
    replay: Buffer,
    idle_replay: usize,
    turn: Turn,
    /// The turn's latest interim text, not yet made final.
    interim: Option<String>,
    /// An `EndOfTurn` arrived for the open turn.
    end_requested: bool,
    end_deadline: Option<Instant>,
    end_hold: Duration,
    end_timeout: Duration,
    events: mpsc::Sender<SttEvent>,
    gone: bool,
    finishing: bool,
}

impl Worker {
    async fn run(mut self, mut input: mpsc::Receiver<SttInput>) {
        while !self.gone {
            let deadline = self.next_deadline();
            tokio::select! {
                biased;
                msg = read(&mut self.active) => self.on_message(msg).await,
                _ = sleep_until(deadline) => self.on_tick().await,
                res = async { self.connecting.as_mut().expect("guarded").await }, if self.connecting.is_some() => {
                    self.connecting = None;
                    self.on_connected(res).await;
                }
                inp = input.recv() => match inp {
                    Some(SttInput::Audio(chunk)) => self.on_audio(&chunk.data).await,
                    Some(SttInput::EndOfTurn) => self.on_end_of_turn().await,
                    None => break,
                },
            }
        }
        if !self.gone {
            self.finish().await;
        }
        if let Some(mut ws) = self.active.take() {
            let _ = ws.close(None).await;
        }
        debug!("gemini live STT: session closed");
    }

    async fn emit(&mut self, ev: SttEvent) {
        if self.events.send(ev).await.is_err() {
            self.gone = true;
        }
    }

    /// The earliest timer: a held turn's end, an `EndOfTurn` timeout, the
    /// gathered audio's send, the connection's rotation.
    fn next_deadline(&self) -> Option<Instant> {
        let hold = match self.turn {
            Turn::Holding { until, .. } => Some(until),
            _ => None,
        };
        let end = self.end_deadline.filter(|_| self.turn == Turn::Open);
        let gather = self.gather_since.map(|t| t + self.chunk);
        let rotate = (!self.rotate_due && !self.finishing && self.active.is_some()).then(|| self.connected_at + self.rotate_after);
        [hold, end, gather, rotate].into_iter().flatten().min()
    }

    async fn on_tick(&mut self) {
        let now = Instant::now();
        if let Turn::Holding { until, reason } = self.turn {
            if until <= now {
                self.close_turn(reason).await;
            }
        }
        if self.turn == Turn::Open && self.end_deadline.is_some_and(|d| d <= now) {
            debug!("gemini live STT: no answer to audioStreamEnd; ending the turn");
            self.close_turn(TurnEnd::Requested).await;
        }
        if self.gather_since.is_some_and(|t| t + self.chunk <= now) {
            self.send_gathered().await;
        }
        if !self.rotate_due && self.active.is_some() && self.connected_at + self.rotate_after <= now {
            debug!("gemini live STT: the connection is due for replacement");
            self.rotate_due = true;
            self.maybe_rotate();
        }
    }

    async fn on_audio(&mut self, pcm: &[u8]) {
        if pcm.is_empty() {
            return;
        }
        if self.gather.is_empty() {
            self.gather_since = Some(Instant::now());
        }
        self.gather.extend_from_slice(pcm);
        if self.gather.len() >= self.chunk_bytes {
            self.send_gathered().await;
        }
    }

    async fn send_gathered(&mut self) {
        self.gather_since = None;
        if self.gather.is_empty() {
            return;
        }
        let pcm = self.gather.split().freeze();
        let Some(ws) = self.active.as_mut() else {
            return self.backlog.push(pcm);
        };
        let sent = ws.send(audio_message(&pcm, &self.mime)).await;
        self.replay.push(pcm);
        if self.turn == Turn::Idle {
            self.replay.trim(self.idle_replay);
        }
        if let Err(e) = sent {
            self.lost(format!("send failed: {e}")).await;
        }
    }

    async fn on_end_of_turn(&mut self) {
        self.send_gathered().await;
        if let Some(ws) = self.active.as_mut() {
            if let Err(e) = ws.send(stream_end()).await {
                self.lost(format!("send failed: {e}")).await;
            }
        }
        if self.turn == Turn::Open && !self.end_requested {
            self.end_requested = true;
            self.end_deadline = Some(Instant::now() + self.end_timeout);
        }
    }

    /// Start opening the next connection, if one is due. It is swapped in at
    /// once when no turn is open, else when the turn ends.
    fn maybe_rotate(&mut self) {
        if self.rotate_due && self.active.is_some() && self.connecting.is_none() && self.standby.is_none() && !self.finishing {
            info!("gemini live STT: opening the next connection");
            self.reconnect();
        }
    }

    async fn on_connected(&mut self, res: Result<Ws>) {
        match res {
            Ok(ws) if self.active.is_none() => self.install(ws).await,
            Ok(ws) => {
                self.failures = 0;
                if self.turn == Turn::Idle {
                    self.swap(ws).await;
                } else {
                    self.standby = Some(ws);
                }
            }
            Err(e) => {
                self.failures += 1;
                if self.active.is_some() {
                    // A rotation failed; the old connection still works.
                    warn!(error = %e, "gemini live STT: could not open the next connection");
                    self.reconnect();
                    return;
                }
                warn!(error = %e, attempt = self.failures, "gemini live STT: reconnect failed");
                if !std::mem::replace(&mut self.outage_reported, true) {
                    self.emit(SttEvent::Error(format!("{LABEL}: reconnect failed: {e:#}"))).await;
                    if self.turn != Turn::Idle {
                        self.replay.clear();
                        self.close_turn(TurnEnd::ConnectionLost).await;
                    }
                }
                self.reconnect();
            }
        }
    }

    /// Make `ws` the connection after a drop: send it the open turn's audio
    /// since its last final, then what arrived meanwhile. A send that fails
    /// keeps the rest and starts another reconnect.
    async fn install(&mut self, mut ws: Ws) {
        info!(replay_bytes = self.replay.bytes, backlog_bytes = self.backlog.bytes, "gemini live STT: reconnected");
        self.connected_at = Instant::now();
        self.rotate_due = false;
        let mut pending = self.replay.take();
        pending.extend(self.backlog.take());
        let mut failed = None;
        while let Some(pcm) = pending.pop_front() {
            if let Err(e) = ws.send(audio_message(&pcm, &self.mime)).await {
                pending.push_front(pcm);
                failed = Some(e);
                break;
            }
            self.replay.push(pcm);
        }
        if failed.is_none() && self.end_requested && self.turn == Turn::Open {
            failed = ws.send(stream_end()).await.err();
        }
        match failed {
            None => {
                self.active = Some(ws);
                self.outage_reported = false;
            }
            Some(e) => {
                warn!(error = %e, "gemini live STT: the new connection failed at once; reconnecting");
                for pcm in pending {
                    self.backlog.push(pcm);
                }
                self.failures += 1;
                self.reconnect();
            }
        }
    }

    /// Replace a working connection by `new` at a turn boundary.
    async fn swap(&mut self, new: Ws) {
        if let Some(mut old) = self.active.replace(new) {
            tokio::spawn(async move {
                let _ = old.send(stream_end()).await;
                let _ = old.close(None).await;
            });
        }
        info!("gemini live STT: switched to the next connection");
        self.connected_at = Instant::now();
        self.rotate_due = false;
        self.replay.clear();
    }

    async fn reconnect_or_standby(&mut self) {
        if let Some(ws) = self.standby.take() {
            self.install(ws).await;
        } else if self.connecting.is_none() {
            self.reconnect();
        }
    }

    fn reconnect(&mut self) {
        let backoff = match self.failures {
            0 => Duration::ZERO,
            n => (Duration::from_millis(200) * 2u32.saturating_pow(n - 1)).min(MAX_BACKOFF),
        };
        let dial = Arc::clone(&self.dial);
        self.connecting = Some(Box::pin(async move {
            tokio::time::sleep(backoff).await;
            dial.connect().await
        }));
    }

    /// The connection is gone. An open turn stays open (its audio is sent
    /// again); a held one ends now.
    async fn lost(&mut self, reason: String) {
        self.active = None;
        if self.finishing {
            debug!(%reason, "gemini live STT: connection ended while finishing");
            return;
        }
        warn!(reason = %redact(&reason), "gemini live STT: connection lost; reconnecting");
        if self.connected_at.elapsed() < STABLE_AFTER {
            self.failures += 1;
        } else {
            self.failures = 0;
        }
        if let Turn::Holding { reason, .. } = self.turn {
            self.close_turn(reason).await;
        }
        self.reconnect_or_standby().await;
    }

    /// Open a turn, ending a held one first.
    async fn start_turn(&mut self) {
        match self.turn {
            Turn::Open => return,
            Turn::Holding { reason, .. } => self.close_turn(reason).await,
            Turn::Idle => {}
        }
        self.turn = Turn::Open;
        self.end_requested = false;
        self.end_deadline = None;
        self.emit(SttEvent::SpeechStarted).await;
    }

    /// The server says the turn is over: end it, or hold it for a final
    /// still missing.
    async fn server_ended_turn(&mut self) {
        let reason = if self.end_requested { TurnEnd::Requested } else { TurnEnd::Silence };
        match self.turn {
            Turn::Idle => {}
            _ if self.interim.is_none() => self.close_turn(reason).await,
            Turn::Open => self.turn = Turn::Holding { until: Instant::now() + self.end_hold, reason },
            Turn::Holding { .. } => {}
        }
    }

    /// End the turn: its last interim, if no final replaced it, becomes final.
    async fn close_turn(&mut self, reason: TurnEnd) {
        if self.turn == Turn::Idle {
            return;
        }
        let reason = if self.finishing { TurnEnd::Closed } else { reason };
        if let Some(text) = self.interim.take() {
            self.emit(SttEvent::Final(text)).await;
        }
        self.turn = Turn::Idle;
        self.end_requested = false;
        self.end_deadline = None;
        self.replay.trim(self.idle_replay);
        self.emit(SttEvent::TurnEnded { reason }).await;
        if let Some(ws) = self.standby.take() {
            self.swap(ws).await;
        }
        self.maybe_rotate();
    }

    async fn on_message(&mut self, msg: Option<Result<Message, tokio_tungstenite::tungstenite::Error>>) {
        let msg = match msg {
            Some(Ok(Message::Close(frame))) => return self.lost(format!("connection closed: {frame:?}")).await,
            Some(Ok(m)) => m,
            Some(Err(e)) => return self.lost(format!("connection error: {e}")).await,
            None => return self.lost("connection closed".to_string()).await,
        };
        let Some(v) = json_of(msg) else { return };
        let activity = v["voiceActivity"]["type"].as_str().unwrap_or("");
        if activity == "ACTIVITY_START" {
            self.start_turn().await;
        }
        let content = &v["serverContent"];
        if let Some(text) = content["interimInputTranscription"]["text"].as_str().filter(|t| !t.is_empty()) {
            if self.turn == Turn::Idle {
                self.start_turn().await;
            }
            self.interim = Some(text.to_string());
            self.emit(SttEvent::Partial(text.to_string())).await;
        }
        if let Some(text) = content["inputTranscription"]["text"].as_str().filter(|t| !t.is_empty()) {
            self.on_final(text.to_string()).await;
        }
        if content["generationComplete"] == true || content["turnComplete"] == true || activity == "ACTIVITY_END" {
            self.server_ended_turn().await;
        }
        if let Some(go) = v.get("goAway") {
            info!(time_left = %go["timeLeft"], "gemini live STT: the server will drop the connection");
            self.rotate_due = true;
            self.maybe_rotate();
        }
        if let Some(err) = v.get("error") {
            let why = err["message"].as_str().unwrap_or("unknown error").to_string();
            warn!(error = %why, "gemini live STT: server error");
            self.emit(SttEvent::Error(format!("{LABEL}: {why}"))).await;
        }
    }

    async fn on_final(&mut self, text: String) {
        self.interim = None;
        match self.turn {
            Turn::Idle => {
                // A final with no turn open: a turn of its own.
                self.turn = Turn::Holding { until: Instant::now() + self.end_hold, reason: TurnEnd::Silence };
                self.emit(SttEvent::Final(text)).await;
            }
            Turn::Open => {
                self.replay.clear();
                self.emit(SttEvent::Final(text)).await;
            }
            Turn::Holding { reason, .. } => {
                // The final the held turn waited for.
                self.emit(SttEvent::Final(text)).await;
                self.close_turn(reason).await;
            }
        }
    }

    /// The input sender is gone: send what is left, end the audio stream, and
    /// take an open turn's last words; it ends with `Closed`.
    async fn finish(&mut self) {
        self.send_gathered().await;
        self.finishing = true;
        self.connecting = None;
        self.standby = None;
        if let Some(ws) = self.active.as_mut() {
            if ws.send(stream_end()).await.is_err() {
                self.active = None;
            }
        }
        if self.turn == Turn::Open {
            self.end_requested = true;
            self.end_deadline = Some(Instant::now() + self.end_timeout.min(FINISH_TIMEOUT));
        }
        while self.turn != Turn::Idle && !self.gone {
            let deadline = self.next_deadline();
            tokio::select! {
                biased;
                msg = read(&mut self.active) => self.on_message(msg).await,
                _ = sleep_until(deadline) => self.on_tick().await,
            }
            if self.active.is_none() {
                break;
            }
        }
        self.close_turn(TurnEnd::Closed).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use tokio::net::TcpListener;

    enum Cmd {
        Send(Value),
        Close,
    }

    /// A fake Live API server on 127.0.0.1. Each connection checks the setup
    /// comes first, answers `setupComplete` in a binary frame (as the real
    /// server may), logs every later message as (connection index, message),
    /// and sends what the test pushes through the command channel handed out
    /// by `next_conn`. While `reject` is set, connections are dropped at once.
    struct FakeServer {
        url: String,
        log: Arc<Mutex<Vec<(usize, Value)>>>,
        setups: Arc<Mutex<Vec<Value>>>,
        conns: mpsc::UnboundedReceiver<mpsc::UnboundedSender<Cmd>>,
        reject: Arc<AtomicBool>,
    }

    impl FakeServer {
        async fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let log = Arc::new(Mutex::new(Vec::new()));
            let setups = Arc::new(Mutex::new(Vec::new()));
            let reject = Arc::new(AtomicBool::new(false));
            let (conn_tx, conns) = mpsc::unbounded_channel();
            let (log2, setups2, reject2) = (Arc::clone(&log), Arc::clone(&setups), Arc::clone(&reject));
            tokio::spawn(async move {
                let mut i = 0;
                loop {
                    let (tcp, _) = listener.accept().await.unwrap();
                    if reject2.load(Ordering::SeqCst) {
                        drop(tcp);
                        continue;
                    }
                    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
                    let _ = conn_tx.send(cmd_tx);
                    tokio::spawn(serve(tcp, i, cmd_rx, Arc::clone(&log2), Arc::clone(&setups2)));
                    i += 1;
                }
            });
            Self { url, log, setups, conns, reject }
        }

        async fn next_conn(&mut self) -> mpsc::UnboundedSender<Cmd> {
            tokio::time::timeout(Duration::from_secs(5), self.conns.recv())
                .await
                .expect("a connection in time")
                .unwrap()
        }

        fn messages(&self, conn: usize) -> Vec<Value> {
            self.log.lock().unwrap().iter().filter(|(i, _)| *i == conn).map(|(_, m)| m.clone()).collect()
        }

        /// Each message of `conn` as `audio:<bytes>` or `end`.
        fn kinds(&self, conn: usize) -> Vec<String> {
            self.messages(conn)
                .iter()
                .map(|m| {
                    let ri = &m["realtimeInput"];
                    if ri["audioStreamEnd"] == true {
                        "end".to_string()
                    } else {
                        assert_eq!(ri["audio"]["mimeType"], "audio/pcm;rate=16000");
                        format!("audio:{}", STANDARD.decode(ri["audio"]["data"].as_str().unwrap()).unwrap().len())
                    }
                })
                .collect()
        }
    }

    async fn serve(
        tcp: tokio::net::TcpStream,
        i: usize,
        mut cmds: mpsc::UnboundedReceiver<Cmd>,
        log: Arc<Mutex<Vec<(usize, Value)>>>,
        setups: Arc<Mutex<Vec<Value>>>,
    ) {
        let mut ws = tokio_tungstenite::accept_hdr_async(
            tcp,
            |req: &tokio_tungstenite::tungstenite::handshake::server::Request, resp| {
                assert_eq!(req.uri().path(), WS_PATH);
                assert_eq!(req.uri().query(), Some("key=k"));
                Ok(resp)
            },
        )
        .await
        .unwrap();
        let Some(Ok(Message::Text(first))) = ws.next().await else { return };
        let setup: Value = serde_json::from_str(&first).unwrap();
        assert!(setup.get("setup").is_some(), "the setup comes first: {setup}");
        setups.lock().unwrap().push(setup);
        ws.send(Message::Binary(json!({"setupComplete": {}}).to_string().into_bytes().into())).await.unwrap();
        loop {
            tokio::select! {
                msg = ws.next() => {
                    let Some(Ok(Message::Text(raw))) = msg else { return };
                    log.lock().unwrap().push((i, serde_json::from_str(&raw).unwrap()));
                }
                cmd = cmds.recv() => match cmd {
                    Some(Cmd::Send(v)) => ws.send(Message::Text(v.to_string().into())).await.unwrap(),
                    Some(Cmd::Close) | None => {
                        let _ = ws.close(None).await;
                        return;
                    }
                },
            }
        }
    }

    fn stt(server: &FakeServer) -> GeminiRealtimeStt {
        GeminiRealtimeStt::new("k").with_base_url(&server.url)
    }

    async fn next(rx: &mut mpsc::Receiver<SttEvent>) -> SttEvent {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event in time")
            .expect("session open")
    }

    /// 20 ms of 16 kHz PCM16.
    fn frame() -> SttInput {
        SttInput::Audio(crate::audio::AudioChunk::new(vec![0u8; 640]))
    }

    fn send(conn: &mpsc::UnboundedSender<Cmd>, v: Value) {
        conn.send(Cmd::Send(v)).unwrap();
    }

    fn interim(t: &str) -> Value {
        json!({"serverContent": {"interimInputTranscription": {"text": t}}})
    }

    fn final_(t: &str) -> Value {
        json!({"serverContent": {"inputTranscription": {"text": t}}})
    }

    fn activity(kind: &str) -> Value {
        json!({"voiceActivity": {"type": kind}})
    }

    fn turn_complete() -> Value {
        json!({"serverContent": {"turnComplete": true}})
    }

    async fn wait_for(cond: impl Fn() -> bool) {
        for _ in 0..300 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("condition never held");
    }

    #[tokio::test]
    async fn setup_comes_first_and_audio_goes_in_100_ms_messages() {
        let mut server = FakeServer::start().await;
        let stt = stt(&server).with_language_codes(["en-US"]).with_custom_vocabulary(["Lumina"]);
        assert_eq!(stt.caps(), SttCaps { ends_turns: true, eager_end: false, sample_rate: 16_000 });
        let (tx, _rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let _conn = server.next_conn().await;
        let setup = server.setups.lock().unwrap()[0].clone();
        assert_eq!(
            setup,
            json!({"setup": {
                "model": "models/gemini-3.5-transcribe-live",
                "generationConfig": {"responseModalities": ["TEXT"]},
                "inputAudioTranscription": {"languageCodes": ["en-US"], "customVocabulary": ["Lumina"]},
            }})
        );
        for _ in 0..7 {
            tx.send(frame()).await.unwrap();
        }
        // Five frames make one 100 ms message; the other two follow on the timer.
        wait_for(|| server.kinds(0).len() == 2).await;
        assert_eq!(server.kinds(0), vec!["audio:3200", "audio:1280"]);
    }

    #[tokio::test]
    async fn a_turn_maps_to_started_partials_finals_and_its_end() {
        let mut server = FakeServer::start().await;
        let (_tx, mut rx) = stt(&server).open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, activity("ACTIVITY_START"));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        send(&conn, interim("run"));
        assert_eq!(next(&mut rx).await, SttEvent::Partial("run".into()));
        send(&conn, interim("run the tests"));
        assert_eq!(next(&mut rx).await, SttEvent::Partial("run the tests".into()));
        send(&conn, final_("Run the tests."));
        assert_eq!(next(&mut rx).await, SttEvent::Final("Run the tests.".into()));
        send(&conn, turn_complete());
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Silence });
        // A second completion signal for the same turn is ignored; an interim
        // with no start signal opens the next turn.
        send(&conn, json!({"serverContent": {"generationComplete": true}}));
        send(&conn, interim("and"));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        assert_eq!(next(&mut rx).await, SttEvent::Partial("and".into()));
    }

    #[tokio::test]
    async fn end_of_turn_sends_audio_stream_end_and_the_turn_ends_requested() {
        let mut server = FakeServer::start().await;
        let (tx, mut rx) = stt(&server).open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, activity("ACTIVITY_START"));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        tx.send(frame()).await.unwrap();
        tx.send(SttInput::EndOfTurn).await.unwrap();
        // The gathered audio goes first, at once.
        wait_for(|| server.kinds(0).len() == 2).await;
        assert_eq!(server.kinds(0), vec!["audio:640", "end"]);
        send(&conn, final_("Done."));
        send(&conn, turn_complete());
        assert_eq!(next(&mut rx).await, SttEvent::Final("Done.".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Requested });
    }

    #[tokio::test]
    async fn a_final_after_the_turn_end_still_belongs_to_it() {
        let mut server = FakeServer::start().await;
        let (_tx, mut rx) = stt(&server).with_end_hold(Duration::from_secs(30)).open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, activity("ACTIVITY_START"));
        send(&conn, interim("hello wor"));
        send(&conn, activity("ACTIVITY_END"));
        send(&conn, final_("Hello world."));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        assert_eq!(next(&mut rx).await, SttEvent::Partial("hello wor".into()));
        assert_eq!(next(&mut rx).await, SttEvent::Final("Hello world.".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Silence });
    }

    #[tokio::test]
    async fn with_no_final_the_last_interim_becomes_final_after_the_hold() {
        let mut server = FakeServer::start().await;
        let (_tx, mut rx) = stt(&server).with_end_hold(Duration::from_millis(50)).open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, interim("never finalized"));
        send(&conn, turn_complete());
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        assert_eq!(next(&mut rx).await, SttEvent::Partial("never finalized".into()));
        assert_eq!(next(&mut rx).await, SttEvent::Final("never finalized".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Silence });
    }

    #[tokio::test]
    async fn a_silent_server_after_end_of_turn_still_ends_the_turn() {
        let mut server = FakeServer::start().await;
        let stt = stt(&server).with_end_timeout(Duration::from_millis(100));
        let (tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, activity("ACTIVITY_START"));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        tx.send(SttInput::EndOfTurn).await.unwrap();
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Requested });
    }

    #[tokio::test]
    async fn a_stray_final_is_a_turn_of_its_own() {
        let mut server = FakeServer::start().await;
        let (_tx, mut rx) = stt(&server).with_end_hold(Duration::from_millis(50)).open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, final_("late"));
        assert_eq!(next(&mut rx).await, SttEvent::Final("late".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Silence });
    }

    #[tokio::test]
    async fn go_away_swaps_in_a_new_connection_at_the_turn_boundary() {
        let mut server = FakeServer::start().await;
        let (tx, mut rx) = stt(&server).with_chunk(Duration::from_millis(20)).open(StreamingSttConfig::default()).await.unwrap();
        let old = server.next_conn().await;
        send(&old, activity("ACTIVITY_START"));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        send(&old, json!({"goAway": {"timeLeft": "50s"}}));
        // Mid-turn: the turn finishes on the old connection.
        let new = server.next_conn().await;
        tx.send(frame()).await.unwrap();
        wait_for(|| server.kinds(0).len() == 1).await;
        send(&old, final_("Still here."));
        send(&old, turn_complete());
        assert_eq!(next(&mut rx).await, SttEvent::Final("Still here.".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Silence });
        // Then the new connection takes the audio; the old one is ended.
        tx.send(frame()).await.unwrap();
        wait_for(|| server.kinds(1).len() == 1).await;
        wait_for(|| server.kinds(0) == vec!["audio:640", "end"]).await;
        send(&new, activity("ACTIVITY_START"));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
    }

    #[tokio::test]
    async fn an_old_connection_is_replaced_while_idle() {
        let mut server = FakeServer::start().await;
        let stt = stt(&server).with_rotate_after(Duration::from_millis(100)).with_chunk(Duration::from_millis(20));
        let (tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let _old = server.next_conn().await;
        let _new = server.next_conn().await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        tx.send(frame()).await.unwrap();
        wait_for(|| server.kinds(1) == vec!["audio:640"]).await;
        assert!(rx.try_recv().is_err(), "a rotation is invisible");
    }

    #[tokio::test]
    async fn a_drop_mid_turn_keeps_the_turn_and_sends_its_audio_again() {
        let mut server = FakeServer::start().await;
        let (tx, mut rx) = stt(&server).with_chunk(Duration::from_millis(20)).open(StreamingSttConfig::default()).await.unwrap();
        let first = server.next_conn().await;
        send(&first, activity("ACTIVITY_START"));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        tx.send(frame()).await.unwrap();
        wait_for(|| server.kinds(0).len() == 1).await;
        send(&first, final_("First part."));
        assert_eq!(next(&mut rx).await, SttEvent::Final("First part.".into()));
        // Audio after the last final: sent again on the next connection.
        for _ in 0..2 {
            tx.send(frame()).await.unwrap();
        }
        wait_for(|| server.kinds(0).len() == 3).await;
        first.send(Cmd::Close).unwrap();
        let second = server.next_conn().await;
        tx.send(frame()).await.unwrap();
        wait_for(|| server.kinds(1).len() == 3).await;
        assert_eq!(server.kinds(1), vec!["audio:640", "audio:640", "audio:640"]);

        send(&second, activity("ACTIVITY_START"));
        send(&second, final_(" second part."));
        send(&second, turn_complete());
        assert_eq!(next(&mut rx).await, SttEvent::Final(" second part.".into()), "the same turn, no new start");
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Silence });
    }

    #[tokio::test]
    async fn a_failed_reconnect_ends_the_open_turn_once() {
        let mut server = FakeServer::start().await;
        let (_tx, mut rx) = stt(&server).open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, activity("ACTIVITY_START"));
        send(&conn, interim("cut o"));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        assert_eq!(next(&mut rx).await, SttEvent::Partial("cut o".into()));
        server.reject.store(true, Ordering::SeqCst);
        conn.send(Cmd::Close).unwrap();
        let SttEvent::Error(e) = next(&mut rx).await else { panic!("error") };
        assert!(e.contains("reconnect failed"), "{e}");
        assert_eq!(next(&mut rx).await, SttEvent::Final("cut o".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::ConnectionLost });
        server.reject.store(false, Ordering::SeqCst);
        let _back = server.next_conn().await;
    }

    #[tokio::test]
    async fn dropping_the_sender_ends_the_stream_and_closes_the_turn() {
        let mut server = FakeServer::start().await;
        let (tx, mut rx) = stt(&server).open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, activity("ACTIVITY_START"));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        tx.send(frame()).await.unwrap();
        drop(tx);
        wait_for(|| server.kinds(0) == vec!["audio:640", "end"]).await;
        send(&conn, final_("bye"));
        send(&conn, turn_complete());
        assert_eq!(next(&mut rx).await, SttEvent::Final("bye".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Closed });
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn open_fails_when_the_server_refuses() {
        let server = FakeServer::start().await;
        server.reject.store(true, Ordering::SeqCst);
        let err = stt(&server).open(StreamingSttConfig::default()).await.unwrap_err();
        assert!(!format!("{err:#}").contains("key=k"), "the key is not in errors: {err:#}");
    }

    #[test]
    fn the_url_carries_the_key_and_errors_hide_it() {
        assert_eq!(
            GeminiRealtimeStt::new("abc").url(),
            format!("wss://generativelanguage.googleapis.com{WS_PATH}?key=abc")
        );
        assert_eq!(redact("failed: wss://h/p?key=AIza-x_1 (io)"), "failed: wss://h/p?key=… (io)");
    }
}
