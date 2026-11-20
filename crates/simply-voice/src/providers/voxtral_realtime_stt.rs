//! Voxtral realtime STT: audio in over one long-lived WebSocket, final text
//! deltas and the provider's turn events out.
//!
//! Protocol (Mistral's *Realtime transcription* and *Turn detection*):
//! connect to `/v1/audio/transcriptions/realtime?model=…` with a Bearer key →
//! `session.created` → one `session.update` with every setting
//! (`audio_format`, `target_streaming_delay_ms`, `turn_detection`) →
//! `session.updated`; then `input_audio.append` (base64 PCM16 LE), answered by
//! `transcription.text.delta` (final as sent, append only) and the turn events
//! `turn.start`, `turn.eager_end`, `turn.resume`, `turn.end` (`end_reason`
//! `threshold` | `timeout`). Settings lock at the first audio chunk, so they
//! all go in the one update, before any audio. `input_audio.flush` and
//! `input_audio.end` finish the audio; `transcription.done` follows.
//!
//! **Mapping.** `turn.start` → [`SttEvent::SpeechStarted`], each delta →
//! [`SttEvent::Final`], `turn.eager_end` / `turn.resume` →
//! [`SttEvent::EagerEnd`] / [`SttEvent::Resumed`], `turn.end` →
//! [`SttEvent::TurnEnded`] ([`TurnEnd::Silence`] or [`TurnEnd::Timeout`]).
//! The last words' text trails `turn.end` by up to the streaming delay, so
//! the turn is held open for one delay after it (or until the next
//! `turn.start`) and the deltas meanwhile still belong to it. Text that
//! arrives with no turn open (late beyond the hold) is a turn of its own,
//! with no `SpeechStarted`, ended one hold later. [`SttInput::EndOfTurn`] is
//! ignored: the provider ends turns.
//!
//! **Reconnect.** A dropped connection (a close, an I/O error, an unexpected
//! `transcription.done`) is replaced in the background, with backoff; audio
//! sent meanwhile is kept (up to [`VoxtralRealtimeStt::with_backlog`]) and
//! sent on the new connection. A turn open at the drop ends at once with
//! [`TurnEnd::ConnectionLost`]: its text so far has already been sent as
//! `Final`s, and the audio in flight at the drop is not transcribed twice.
//! A reconnect that fails is reported once per outage as
//! [`SttEvent::Error`].

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, info, warn};

use super::voxtral::{DEFAULT_BASE_URL, DEFAULT_STT_MODEL};
use super::voxtral_ws::{self, error_message, Ws};
use crate::provider::streaming_stt::SESSION_CHANNEL;
use crate::provider::{SttCaps, SttEvent, SttInput, SttSession, StreamingStt, StreamingSttConfig, TurnEnd};

/// Names this endpoint in errors.
const LABEL: &str = "voxtral realtime STT";

/// How long the last words may take to arrive after the input sender is
/// dropped.
const FINISH_TIMEOUT: Duration = Duration::from_secs(5);
/// The longest wait between two reconnect attempts.
const MAX_BACKOFF: Duration = Duration::from_secs(5);
/// A connection lost sooner than this after it opened counts as a failed
/// attempt, so a server that accepts and drops at once is retried with
/// backoff.
const STABLE_AFTER: Duration = Duration::from_secs(2);

/// Mistral's turn detection settings. See *Turn detection* in Mistral's docs:
/// the thresholds must keep `start_threshold ≥ eager_end_threshold >
/// end_threshold`, and `end_timeout_ms ≥ min_silence_ms`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct VoxtralTurnDetection {
    /// Score above which the user starts or resumes speaking.
    pub start_threshold: f32,
    /// Score below which the user may have stopped (an eager end).
    pub eager_end_threshold: f32,
    /// Score below which the user has stopped, once `min_silence_ms` passed.
    pub end_threshold: f32,
    /// Shortest pause that can end a turn.
    pub min_silence_ms: u32,
    /// Longest pause before the turn ends anyway (`end_reason: timeout`).
    pub end_timeout_ms: u32,
}

impl VoxtralTurnDetection {
    /// Mistral's "Aggressive" values: the turn can end 300 ms into a pause.
    pub fn aggressive() -> Self {
        Self {
            start_threshold: 0.5,
            eager_end_threshold: 0.4,
            end_threshold: 0.33,
            min_silence_ms: 300,
            end_timeout_ms: 2500,
        }
    }

    /// The server's defaults: 500 ms minimum pause, 3 s timeout.
    pub fn mistral_default() -> Self {
        Self {
            start_threshold: 0.5,
            eager_end_threshold: 0.5,
            end_threshold: 0.33,
            min_silence_ms: 500,
            end_timeout_ms: 3000,
        }
    }

    fn to_json(self) -> Value {
        json!({
            "start_threshold": self.start_threshold,
            "eager_end_threshold": self.eager_end_threshold,
            "end_threshold": self.end_threshold,
            "min_silence_ms": self.min_silence_ms,
            "end_timeout_ms": self.end_timeout_ms,
        })
    }
}

impl Default for VoxtralTurnDetection {
    /// [`VoxtralTurnDetection::aggressive`].
    fn default() -> Self {
        Self::aggressive()
    }
}

/// Voxtral's realtime transcription behind [`StreamingStt`], with the
/// provider's turn detection. Takes PCM16 at any sample rate the session
/// declares; 16 kHz is preferred.
#[derive(Debug, Clone)]
pub struct VoxtralRealtimeStt {
    api_key: String,
    base_url: String,
    model: String,
    turn_detection: VoxtralTurnDetection,
    streaming_delay: Duration,
    end_hold: Option<Duration>,
    handshake_timeout: Duration,
    backlog: Duration,
}

impl VoxtralRealtimeStt {
    /// Mistral's hosted API, model `voxtral-mini-transcribe-realtime-3`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_STT_MODEL.to_string(),
            turn_detection: VoxtralTurnDetection::default(),
            streaming_delay: Duration::from_millis(240),
            end_hold: None,
            handshake_timeout: Duration::from_secs(10),
            backlog: Duration::from_secs(10),
        }
    }

    /// A local or compatible server at `base_url` (`http://…` or `https://…`),
    /// with no key.
    pub fn local(base_url: impl Into<String>) -> Self {
        Self::new(String::new()).with_base_url(base_url)
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Turn detection settings; [`VoxtralTurnDetection::aggressive`] by default.
    pub fn with_turn_detection(mut self, turn_detection: VoxtralTurnDetection) -> Self {
        self.turn_detection = turn_detection;
        self
    }

    /// `target_streaming_delay_ms`: how far the text runs behind the audio.
    /// A multiple of 80 ms from 80 to 1200 ms, or 2400 ms; 240 ms by default,
    /// at most the minimum silence so the hold after a turn end is mostly
    /// empty. Another value fails `open`.
    pub fn with_streaming_delay(mut self, delay: Duration) -> Self {
        self.streaming_delay = delay;
        self
    }

    /// How long a turn stays open after `turn.end` for its last words; the
    /// streaming delay by default.
    pub fn with_end_hold(mut self, hold: Duration) -> Self {
        self.end_hold = Some(hold);
        self
    }

    /// How long connecting and configuring a connection may take; 10 s by
    /// default.
    pub fn with_handshake_timeout(mut self, timeout: Duration) -> Self {
        self.handshake_timeout = timeout;
        self
    }

    /// How much audio is kept while reconnecting; the oldest goes first.
    /// 10 s by default.
    pub fn with_backlog(mut self, backlog: Duration) -> Self {
        self.backlog = backlog;
        self
    }

    fn url(&self) -> String {
        voxtral_ws::ws_url(&self.base_url, &format!("/v1/audio/transcriptions/realtime?model={}", self.model))
    }

    fn delay_ms(&self) -> Result<u64> {
        let ms = self.streaming_delay.as_millis() as u64;
        if (ms.is_multiple_of(80) && (80..=1200).contains(&ms)) || ms == 2400 {
            Ok(ms)
        } else {
            anyhow::bail!("{LABEL}: streaming delay {ms} ms is not a multiple of 80 from 80 to 1200, or 2400")
        }
    }
}

#[async_trait]
impl StreamingStt for VoxtralRealtimeStt {
    fn caps(&self) -> SttCaps {
        SttCaps { ends_turns: true, eager_end: true, sample_rate: crate::audio::SAMPLE_RATE }
    }

    async fn open(&self, cfg: StreamingSttConfig) -> Result<SttSession> {
        let delay_ms = self.delay_ms()?;
        let update = json!({
            "type": "session.update",
            "session": {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": cfg.sample_rate},
                "target_streaming_delay_ms": delay_ms,
                "turn_detection": self.turn_detection.to_json(),
            },
        });
        let dial = Arc::new(Dial {
            url: self.url(),
            api_key: self.api_key.clone(),
            update: update.to_string(),
            timeout: self.handshake_timeout,
        });
        info!(url = %dial.url, "voxtral realtime STT: opening session");
        let first = dial.connect().await?;

        let (in_tx, in_rx) = mpsc::channel(SESSION_CHANNEL);
        let (ev_tx, ev_rx) = mpsc::channel(SESSION_CHANNEL);
        let worker = Worker {
            dial,
            active: Some(first),
            connected_at: Instant::now(),
            connecting: None,
            failures: 0,
            backlog: VecDeque::new(),
            backlog_bytes: 0,
            backlog_cap: (self.backlog.as_secs_f64() * cfg.sample_rate as f64 * 2.0) as usize,
            turn: Turn::Idle,
            hold: self.end_hold.unwrap_or(Duration::from_millis(delay_ms)),
            events: ev_tx,
            gone: false,
            finishing: false,
        };
        tokio::spawn(worker.run(in_rx));
        Ok((in_tx, ev_rx))
    }
}

/// How to open and configure one connection.
struct Dial {
    url: String,
    api_key: String,
    /// The `session.update` message, sent on every connection before audio.
    update: String,
    timeout: Duration,
}

impl Dial {
    async fn connect(&self) -> Result<Ws> {
        tokio::time::timeout(self.timeout, self.handshake())
            .await
            .map_err(|_| anyhow::anyhow!("{LABEL}: handshake timed out"))?
    }

    async fn handshake(&self) -> Result<Ws> {
        let mut ws = voxtral_ws::connect(&self.url, &self.api_key)
            .await
            .context("voxtral realtime STT: connect")?;
        voxtral_ws::expect(&mut ws, "session.created", LABEL).await?;
        ws.send(Message::Text(self.update.clone().into())).await?;
        let applied = voxtral_ws::expect(&mut ws, "session.updated", LABEL).await?;
        debug!(session = %applied["session"], "voxtral realtime STT: connection ready");
        Ok(ws)
    }
}

/// Where the current turn is.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Turn {
    Idle,
    /// Between `turn.start` and `turn.end`.
    Open,
    /// After `turn.end` (or stray text), until `until`: then it ends with
    /// `reason`.
    Holding { until: Instant, reason: TurnEnd },
}

type Connecting = Pin<Box<dyn Future<Output = Result<Ws>> + Send>>;

/// The next message on `active`, or never when there is none.
async fn read(active: &mut Option<Ws>) -> Option<Result<Message, tokio_tungstenite::tungstenite::Error>> {
    match active {
        Some(ws) => ws.next().await,
        None => std::future::pending().await,
    }
}

async fn hold_expiry(turn: Turn) {
    match turn {
        Turn::Holding { until, .. } => tokio::time::sleep_until(until).await,
        _ => std::future::pending().await,
    }
}

fn append(pcm: &[u8]) -> Message {
    Message::Text(json!({"type": "input_audio.append", "audio": STANDARD.encode(pcm)}).to_string().into())
}

fn control(kind: &str) -> Message {
    Message::Text(json!({"type": kind}).to_string().into())
}

/// One session: owns the connection, reconnects it, and assembles turns.
struct Worker {
    dial: Arc<Dial>,
    active: Option<Ws>,
    /// When `active` opened.
    connected_at: Instant,
    /// A reconnect in progress (backoff included).
    connecting: Option<Connecting>,
    /// Failed reconnect attempts in this outage.
    failures: u32,
    /// Audio received while there was no connection, oldest first.
    backlog: VecDeque<Bytes>,
    backlog_bytes: usize,
    backlog_cap: usize,
    turn: Turn,
    hold: Duration,
    events: mpsc::Sender<SttEvent>,
    /// The event receiver was dropped.
    gone: bool,
    /// The input sender was dropped: a lost connection is not replaced.
    finishing: bool,
}

impl Worker {
    async fn run(mut self, mut input: mpsc::Receiver<SttInput>) {
        while !self.gone {
            let turn = self.turn;
            tokio::select! {
                biased;
                msg = read(&mut self.active) => self.on_message(msg).await,
                _ = hold_expiry(turn) => self.end_held_turn().await,
                res = async { self.connecting.as_mut().expect("guarded").await }, if self.connecting.is_some() => {
                    self.connecting = None;
                    self.on_connected(res).await;
                }
                inp = input.recv() => match inp {
                    Some(SttInput::Audio(chunk)) => self.on_audio(chunk.data).await,
                    Some(SttInput::EndOfTurn) => debug!("voxtral realtime STT: EndOfTurn ignored, the provider ends turns"),
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
        debug!("voxtral realtime STT: session closed");
    }

    async fn emit(&mut self, ev: SttEvent) {
        if self.events.send(ev).await.is_err() {
            self.gone = true;
        }
    }

    async fn on_audio(&mut self, pcm: Bytes) {
        if pcm.is_empty() {
            return;
        }
        if let Some(ws) = self.active.as_mut() {
            match ws.send(append(&pcm)).await {
                Ok(()) => return,
                Err(e) => {
                    self.keep(pcm);
                    return self.lost(format!("send failed: {e}")).await;
                }
            }
        }
        self.keep(pcm);
    }

    /// Keep audio for the next connection, dropping the oldest past the cap.
    fn keep(&mut self, pcm: Bytes) {
        self.backlog_bytes += pcm.len();
        self.backlog.push_back(pcm);
        while self.backlog_bytes > self.backlog_cap {
            match self.backlog.pop_front() {
                Some(old) => self.backlog_bytes -= old.len(),
                None => break,
            }
        }
    }

    async fn on_connected(&mut self, res: Result<Ws>) {
        match res {
            Ok(mut ws) => {
                info!(backlog_bytes = self.backlog_bytes, "voxtral realtime STT: reconnected");
                self.connected_at = Instant::now();
                while let Some(pcm) = self.backlog.pop_front() {
                    self.backlog_bytes -= pcm.len();
                    if let Err(e) = ws.send(append(&pcm)).await {
                        self.keep(pcm);
                        return self.lost(format!("send failed: {e}")).await;
                    }
                }
                self.active = Some(ws);
            }
            Err(e) => {
                self.failures += 1;
                warn!(error = %e, attempt = self.failures, "voxtral realtime STT: reconnect failed");
                if self.failures == 1 {
                    self.emit(SttEvent::Error(format!("{LABEL}: reconnect failed: {e:#}"))).await;
                }
                self.reconnect();
            }
        }
    }

    /// Start a reconnect, after a backoff that doubles with each failure.
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

    /// The connection is gone. An open turn ends now (a held one with the
    /// provider's reason); a new connection is opened unless the session is
    /// finishing.
    async fn lost(&mut self, reason: String) {
        self.active = None;
        if self.finishing {
            debug!(%reason, "voxtral realtime STT: connection ended while finishing");
            return;
        }
        warn!(%reason, "voxtral realtime STT: connection lost; reconnecting");
        if self.connected_at.elapsed() < STABLE_AFTER {
            self.failures += 1;
        } else {
            self.failures = 0;
        }
        match self.turn {
            Turn::Idle => {}
            Turn::Holding { .. } => self.end_held_turn().await,
            Turn::Open => {
                self.turn = Turn::Idle;
                self.emit(SttEvent::TurnEnded { reason: TurnEnd::ConnectionLost }).await;
            }
        }
        if self.connecting.is_none() {
            self.reconnect();
        }
    }

    async fn end_held_turn(&mut self) {
        if let Turn::Holding { reason, .. } = self.turn {
            self.turn = Turn::Idle;
            self.emit(SttEvent::TurnEnded { reason }).await;
        }
    }

    async fn on_message(&mut self, msg: Option<Result<Message, tokio_tungstenite::tungstenite::Error>>) {
        let text = match msg {
            Some(Ok(Message::Text(text))) => text,
            Some(Ok(Message::Close(frame))) => return self.lost(format!("connection closed: {frame:?}")).await,
            Some(Ok(_)) => return,
            Some(Err(e)) => return self.lost(format!("connection error: {e}")).await,
            None => return self.lost("connection closed".to_string()).await,
        };
        let msg: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "voxtral realtime STT: unparsable event");
                return;
            }
        };
        match msg["type"].as_str().unwrap_or("") {
            "transcription.text.delta" => {
                let Some(t) = msg["text"].as_str().filter(|t| !t.is_empty()) else { return };
                if self.turn == Turn::Idle {
                    // Text with no turn open: a turn of its own.
                    self.turn = Turn::Holding { until: Instant::now() + self.hold, reason: TurnEnd::Silence };
                }
                self.emit(SttEvent::Final(t.to_string())).await;
            }
            "turn.start" => {
                match self.turn {
                    Turn::Open => return,
                    Turn::Holding { .. } => self.end_held_turn().await,
                    Turn::Idle => {}
                }
                self.turn = Turn::Open;
                self.emit(SttEvent::SpeechStarted).await;
            }
            "turn.eager_end" if self.turn == Turn::Open => self.emit(SttEvent::EagerEnd).await,
            "turn.resume" if self.turn == Turn::Open => self.emit(SttEvent::Resumed).await,
            "turn.end" if self.turn == Turn::Open => {
                let reason = match msg["end_reason"].as_str() {
                    Some("timeout") => TurnEnd::Timeout,
                    _ => TurnEnd::Silence,
                };
                self.turn = Turn::Holding { until: Instant::now() + self.hold, reason };
            }
            "transcription.done" => {
                if self.finishing {
                    self.active = None;
                } else {
                    self.lost("the server ended the transcription".to_string()).await;
                }
            }
            "error" => {
                let reason = error_message(&msg);
                warn!(error = %reason, "voxtral realtime STT: server error");
                self.emit(SttEvent::Error(format!("{LABEL}: {reason}"))).await;
            }
            other => debug!(event = %other, "voxtral realtime STT: ignored event"),
        }
    }

    /// The input sender is gone: send what is left, end the audio, and take
    /// the last words until `transcription.done`; then end an open turn.
    async fn finish(&mut self) {
        self.finishing = true;
        self.connecting = None;
        if let Some(ws) = self.active.as_mut() {
            let sent = async {
                ws.send(control("input_audio.flush")).await?;
                ws.send(control("input_audio.end")).await
            };
            if let Err(e) = sent.await {
                debug!(error = %e, "voxtral realtime STT: could not end the audio");
                self.active = None;
            }
        }
        let deadline = tokio::time::sleep(FINISH_TIMEOUT);
        tokio::pin!(deadline);
        while self.active.is_some() && !self.gone {
            let turn = self.turn;
            tokio::select! {
                biased;
                msg = read(&mut self.active) => self.on_message(msg).await,
                _ = hold_expiry(turn) => self.end_held_turn().await,
                _ = &mut deadline => break,
            }
        }
        match self.turn {
            Turn::Idle => {}
            Turn::Open => self.emit(SttEvent::TurnEnded { reason: TurnEnd::Closed }).await,
            Turn::Holding { .. } => self.end_held_turn().await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tokio::net::TcpListener;

    /// What the test tells a fake connection to do.
    enum Cmd {
        Send(Value),
        Close,
    }

    /// A fake Mistral realtime STT server on 127.0.0.1. Each connection
    /// answers the handshake, logs every message it receives as (connection
    /// index, message), and sends what the test pushes through its command
    /// channel, handed out by `next_conn`.
    struct FakeServer {
        url: String,
        log: Arc<Mutex<Vec<(usize, Value)>>>,
        conns: mpsc::UnboundedReceiver<mpsc::UnboundedSender<Cmd>>,
    }

    impl FakeServer {
        async fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let log = Arc::new(Mutex::new(Vec::new()));
            let (conn_tx, conns) = mpsc::unbounded_channel();
            let log2 = Arc::clone(&log);
            tokio::spawn(async move {
                for i in 0.. {
                    let (tcp, _) = listener.accept().await.unwrap();
                    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
                    let _ = conn_tx.send(cmd_tx);
                    tokio::spawn(serve(tcp, i, cmd_rx, Arc::clone(&log2)));
                }
            });
            Self { url, log, conns }
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

        fn types(&self, conn: usize) -> Vec<String> {
            self.messages(conn).iter().map(|m| m["type"].as_str().unwrap().to_string()).collect()
        }
    }

    async fn serve(
        tcp: tokio::net::TcpStream,
        i: usize,
        mut cmds: mpsc::UnboundedReceiver<Cmd>,
        log: Arc<Mutex<Vec<(usize, Value)>>>,
    ) {
        let mut ws = tokio_tungstenite::accept_hdr_async(
            tcp,
            |req: &tokio_tungstenite::tungstenite::handshake::server::Request, resp| {
                assert!(req.uri().path().ends_with("/v1/audio/transcriptions/realtime"));
                assert!(req.uri().query().unwrap().contains("model="));
                Ok(resp)
            },
        )
        .await
        .unwrap();
        let text = |v: Value| Message::Text(v.to_string().into());
        ws.send(text(json!({"type": "session.created", "session": {}}))).await.unwrap();
        loop {
            tokio::select! {
                msg = ws.next() => {
                    let Some(Ok(Message::Text(raw))) = msg else { return };
                    let m: Value = serde_json::from_str(&raw).unwrap();
                    log.lock().unwrap().push((i, m.clone()));
                    if m["type"] == "session.update" {
                        ws.send(text(json!({"type": "session.updated", "session": m["session"]}))).await.unwrap();
                    }
                }
                cmd = cmds.recv() => match cmd {
                    Some(Cmd::Send(v)) => ws.send(text(v)).await.unwrap(),
                    Some(Cmd::Close) | None => {
                        let _ = ws.close(None).await;
                        return;
                    }
                },
            }
        }
    }

    async fn next(rx: &mut mpsc::Receiver<SttEvent>) -> SttEvent {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event in time")
            .expect("session open")
    }

    fn audio(n: usize) -> SttInput {
        SttInput::Audio(crate::audio::AudioChunk::new(vec![0u8; n]))
    }

    fn send(conn: &mpsc::UnboundedSender<Cmd>, v: Value) {
        conn.send(Cmd::Send(v)).unwrap();
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
    async fn the_session_is_configured_before_any_audio() {
        let mut server = FakeServer::start().await;
        let stt = VoxtralRealtimeStt::local(&server.url);
        let caps = stt.caps();
        assert!(caps.ends_turns && caps.eager_end);
        let (tx, _rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let _conn = server.next_conn().await;
        tx.send(audio(640)).await.unwrap();
        tx.send(audio(640)).await.unwrap();
        wait_for(|| server.messages(0).len() == 3).await;

        assert_eq!(server.types(0), vec!["session.update", "input_audio.append", "input_audio.append"]);
        let update = &server.messages(0)[0]["session"];
        assert_eq!(update["audio_format"], json!({"encoding": "pcm_s16le", "sample_rate": 16000}));
        assert_eq!(update["target_streaming_delay_ms"], 240);
        let td = &update["turn_detection"];
        assert_eq!(td["min_silence_ms"], 300);
        assert_eq!(td["end_timeout_ms"], 2500);
        assert_eq!(td["eager_end_threshold"].as_f64().unwrap() as f32, 0.4);
        assert_eq!(td["start_threshold"], 0.5);
        let audio = STANDARD.decode(server.messages(0)[1]["audio"].as_str().unwrap()).unwrap();
        assert_eq!(audio.len(), 640);
    }

    #[tokio::test]
    async fn a_turn_maps_to_events_and_holds_open_for_its_last_words() {
        let mut server = FakeServer::start().await;
        let stt = VoxtralRealtimeStt::local(&server.url).with_streaming_delay(Duration::from_millis(240));
        let (_tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;

        send(&conn, json!({"type": "turn.start", "time_s": 0.1}));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        send(&conn, json!({"type": "transcription.text.delta", "text": "Run the"}));
        assert_eq!(next(&mut rx).await, SttEvent::Final("Run the".into()));
        send(&conn, json!({"type": "turn.eager_end", "time_s": 0.9}));
        assert_eq!(next(&mut rx).await, SttEvent::EagerEnd);
        send(&conn, json!({"type": "turn.resume", "time_s": 1.0}));
        assert_eq!(next(&mut rx).await, SttEvent::Resumed);

        send(&conn, json!({"type": "turn.end", "time_s": 1.6, "end_reason": "threshold"}));
        let ended_at = Instant::now();
        // The last words trail the event: still this turn's.
        send(&conn, json!({"type": "transcription.text.delta", "text": " tests."}));
        assert_eq!(next(&mut rx).await, SttEvent::Final(" tests.".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Silence });
        let held = ended_at.elapsed();
        assert!(held >= Duration::from_millis(200), "held for the streaming delay, got {held:?}");
    }

    #[tokio::test]
    async fn the_next_turn_start_ends_a_held_turn_at_once() {
        let mut server = FakeServer::start().await;
        let stt = VoxtralRealtimeStt::local(&server.url).with_end_hold(Duration::from_secs(30));
        let (_tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;

        send(&conn, json!({"type": "turn.start", "time_s": 0.0}));
        send(&conn, json!({"type": "transcription.text.delta", "text": "Hmm"}));
        send(&conn, json!({"type": "turn.end", "time_s": 2.5, "end_reason": "timeout"}));
        send(&conn, json!({"type": "turn.start", "time_s": 2.7}));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        assert_eq!(next(&mut rx).await, SttEvent::Final("Hmm".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Timeout });
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
    }

    #[tokio::test]
    async fn text_outside_a_turn_is_a_turn_of_its_own() {
        let mut server = FakeServer::start().await;
        let stt = VoxtralRealtimeStt::local(&server.url).with_end_hold(Duration::from_millis(50));
        let (_tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, json!({"type": "transcription.text.delta", "text": "late"}));
        assert_eq!(next(&mut rx).await, SttEvent::Final("late".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Silence });
    }

    #[tokio::test]
    async fn a_dropped_connection_ends_the_open_turn_and_reconnects() {
        let mut server = FakeServer::start().await;
        let stt = VoxtralRealtimeStt::local(&server.url);
        let (tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let first = server.next_conn().await;

        send(&first, json!({"type": "turn.start", "time_s": 0.0}));
        send(&first, json!({"type": "transcription.text.delta", "text": "half a"}));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        assert_eq!(next(&mut rx).await, SttEvent::Final("half a".into()));
        first.send(Cmd::Close).unwrap();
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::ConnectionLost });

        // Audio sent across the reconnect reaches the new connection, after
        // its session.update.
        tx.send(audio(320)).await.unwrap();
        let second = server.next_conn().await;
        tx.send(audio(320)).await.unwrap();
        wait_for(|| server.types(1).iter().filter(|t| *t == "input_audio.append").count() == 2).await;
        assert_eq!(server.types(1)[0], "session.update");

        send(&second, json!({"type": "turn.start", "time_s": 0.0}));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        send(&second, json!({"type": "transcription.text.delta", "text": "again"}));
        assert_eq!(next(&mut rx).await, SttEvent::Final("again".into()));
    }

    #[tokio::test]
    async fn a_drop_during_the_hold_ends_the_turn_with_the_providers_reason() {
        let mut server = FakeServer::start().await;
        let stt = VoxtralRealtimeStt::local(&server.url).with_end_hold(Duration::from_secs(30));
        let (_tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, json!({"type": "turn.start", "time_s": 0.0}));
        send(&conn, json!({"type": "turn.end", "time_s": 1.0, "end_reason": "threshold"}));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        conn.send(Cmd::Close).unwrap();
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Silence });
        let _second = server.next_conn().await;
    }

    #[tokio::test]
    async fn end_of_turn_is_ignored() {
        let mut server = FakeServer::start().await;
        let stt = VoxtralRealtimeStt::local(&server.url);
        let (tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, json!({"type": "turn.start", "time_s": 0.0}));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        tx.send(SttInput::EndOfTurn).await.unwrap();
        tx.send(audio(2)).await.unwrap();
        wait_for(|| server.types(0).len() == 2).await;
        assert_eq!(server.types(0), vec!["session.update", "input_audio.append"]);
        assert!(rx.try_recv().is_err(), "no event for EndOfTurn");
    }

    #[tokio::test]
    async fn dropping_the_sender_takes_the_last_words_then_closes() {
        let mut server = FakeServer::start().await;
        let stt = VoxtralRealtimeStt::local(&server.url);
        let (tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, json!({"type": "turn.start", "time_s": 0.0}));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
        tx.send(audio(2)).await.unwrap();
        drop(tx);
        wait_for(|| server.types(0).last().map(String::as_str) == Some("input_audio.end")).await;
        assert_eq!(server.types(0), vec!["session.update", "input_audio.append", "input_audio.flush", "input_audio.end"]);
        send(&conn, json!({"type": "transcription.text.delta", "text": "bye"}));
        send(&conn, json!({"type": "transcription.done", "text": "bye"}));
        assert_eq!(next(&mut rx).await, SttEvent::Final("bye".into()));
        assert_eq!(next(&mut rx).await, SttEvent::TurnEnded { reason: TurnEnd::Closed });
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn a_server_error_is_reported_and_the_session_goes_on() {
        let mut server = FakeServer::start().await;
        let stt = VoxtralRealtimeStt::local(&server.url);
        let (_tx, mut rx) = stt.open(StreamingSttConfig::default()).await.unwrap();
        let conn = server.next_conn().await;
        send(&conn, json!({"type": "error", "error": {"message": "bad chunk", "code": 400}}));
        let SttEvent::Error(e) = next(&mut rx).await else { panic!("error") };
        assert!(e.contains("bad chunk"), "{e}");
        send(&conn, json!({"type": "turn.start", "time_s": 0.0}));
        assert_eq!(next(&mut rx).await, SttEvent::SpeechStarted);
    }

    #[tokio::test]
    async fn open_fails_on_a_bad_delay_or_an_unreachable_server() {
        let stt = VoxtralRealtimeStt::local("http://127.0.0.1:9").with_streaming_delay(Duration::from_millis(250));
        let err = stt.open(StreamingSttConfig::default()).await.unwrap_err();
        assert!(format!("{err:#}").contains("250 ms"), "{err:#}");

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        assert!(VoxtralRealtimeStt::local(url).open(StreamingSttConfig::default()).await.is_err());
    }

    #[test]
    fn the_url_follows_the_base_url_scheme() {
        assert_eq!(
            VoxtralRealtimeStt::new("k").url(),
            "wss://api.mistral.ai/v1/audio/transcriptions/realtime?model=voxtral-mini-transcribe-realtime-3"
        );
        assert_eq!(
            VoxtralRealtimeStt::local("http://localhost:8000/").with_model("m").url(),
            "ws://localhost:8000/v1/audio/transcriptions/realtime?model=m"
        );
    }
}
