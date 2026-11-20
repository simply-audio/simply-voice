//! Voxtral realtime TTS: text deltas in over a WebSocket, audio deltas out,
//! one long-lived connection per session.
//!
//! Protocol (Mistral's *Speak in realtime*): connect to
//! `/v1/audio/speech/realtime?model=…` with a Bearer key → `session.created`
//! → `session.update` (voice, `streaming_mode`, `response_format`) →
//! `session.updated`; then per generation `input_text.append` deltas and an
//! `input_text.flush`, answered by `speech.audio.delta` (base64 `audio_data`,
//! possibly before the flush) and `speech.audio.done`. Generations queue in
//! order on one connection; a failed generation is an `error` event and closes
//! the connection.
//!
//! There is no cancel message, so `Cancel` closes the busy connection and
//! swaps in a spare one opened in advance: the session stays usable and the
//! next utterance pays no handshake. A lost connection fails the generation in
//! progress; the utterances queued behind it are sent again on a new one.
//!
//! **Audio.** `response_format` is `pcm`: raw mono 24 kHz samples, float32
//! little-endian (the same as the REST endpoint's `pcm`). Deltas are not cut on
//! sample boundaries, so bytes are carried over to keep every
//! [`TtsEvent::Audio`] whole samples. Mistral's docs say mono 24 kHz but not
//! the sample type; a session whose samples don't look like float32 logs a
//! warning.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use futures::FutureExt;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::{HeaderValue, AUTHORIZATION};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tracing::{debug, info, warn};

use super::voxtral::{DEFAULT_BASE_URL, DEFAULT_TTS_MODEL};
use crate::audio::{AudioChunk, AudioEncoding, AudioFormat};
use crate::provider::streaming_tts::SESSION_CHANNEL;
use crate::provider::{StreamingTts, StreamingTtsConfig, TtsEvent, TtsInput, TtsSession};

/// What every session's audio is: float32 LE, mono, 24 kHz.
pub const VOXTRAL_REALTIME_FORMAT: AudioFormat =
    AudioFormat { encoding: AudioEncoding::F32, sample_rate: 24_000, channels: 1 };

/// Voxtral's latency/quality trade-off for a realtime session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VoxtralStreamingMode {
    /// Lower latency; Mistral's default.
    #[default]
    Responsive,
    /// Higher quality for longer text.
    LongForm,
}

impl VoxtralStreamingMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Responsive => "responsive",
            Self::LongForm => "long_form",
        }
    }
}

/// Voxtral's realtime TTS behind [`StreamingTts`]. Audio is
/// [`VOXTRAL_REALTIME_FORMAT`].
#[derive(Debug, Clone)]
pub struct VoxtralRealtimeTts {
    api_key: String,
    base_url: String,
    model: String,
    streaming_mode: VoxtralStreamingMode,
    handshake_timeout: Duration,
}

impl VoxtralRealtimeTts {
    /// Mistral's hosted API.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_TTS_MODEL.to_string(),
            streaming_mode: VoxtralStreamingMode::default(),
            handshake_timeout: Duration::from_secs(10),
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

    pub fn with_streaming_mode(mut self, mode: VoxtralStreamingMode) -> Self {
        self.streaming_mode = mode;
        self
    }

    /// How long connecting and configuring a connection may take.
    pub fn with_handshake_timeout(mut self, timeout: Duration) -> Self {
        self.handshake_timeout = timeout;
        self
    }

    fn url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        let ws = if let Some(rest) = base.strip_prefix("https://") {
            format!("wss://{rest}")
        } else if let Some(rest) = base.strip_prefix("http://") {
            format!("ws://{rest}")
        } else {
            base.to_string()
        };
        format!("{ws}/v1/audio/speech/realtime?model={}", self.model)
    }
}

#[async_trait]
impl StreamingTts for VoxtralRealtimeTts {
    async fn open(&self, cfg: StreamingTtsConfig) -> Result<TtsSession> {
        let mut session = json!({
            "streaming_mode": self.streaming_mode.as_str(),
            "response_format": "pcm",
        });
        if !cfg.voice.is_empty() {
            session["voice"] = json!(cfg.voice);
        }
        let dial = Arc::new(Dial {
            url: self.url(),
            api_key: self.api_key.clone(),
            update: json!({"type": "session.update", "session": session}).to_string(),
            timeout: self.handshake_timeout,
        });
        info!(url = %dial.url, "voxtral realtime TTS: opening session");
        let first = dial.connect().await?;

        let (in_tx, in_rx) = mpsc::channel(SESSION_CHANNEL);
        let (ev_tx, ev_rx) = mpsc::channel(SESSION_CHANNEL);
        let worker = Worker {
            spare: Some(dial.spawn()),
            dial,
            active: Some(first),
            current: String::new(),
            current_failed: None,
            pending: VecDeque::new(),
            remainder: Vec::new(),
            events: ev_tx,
            gone: false,
            checked_format: false,
        };
        tokio::spawn(worker.run(in_rx));
        Ok((VOXTRAL_REALTIME_FORMAT, in_tx, ev_rx))
    }
}

type Ws = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// How to open and configure one connection.
struct Dial {
    url: String,
    api_key: String,
    /// The `session.update` message, sent on every connection.
    update: String,
    timeout: Duration,
}

impl Dial {
    async fn connect(&self) -> Result<Ws> {
        tokio::time::timeout(self.timeout, self.handshake())
            .await
            .map_err(|_| anyhow::anyhow!("voxtral realtime TTS: handshake timed out"))?
    }

    async fn handshake(&self) -> Result<Ws> {
        let mut req = self.url.as_str().into_client_request()?;
        if !self.api_key.is_empty() {
            req.headers_mut()
                .insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", self.api_key))?);
        }
        let (mut ws, _) = connect_async(req).await.context("voxtral realtime TTS: connect")?;
        expect(&mut ws, "session.created").await?;
        ws.send(Message::Text(self.update.clone().into())).await?;
        expect(&mut ws, "session.updated").await?;
        debug!("voxtral realtime TTS: connection ready");
        Ok(ws)
    }

    fn spawn(self: &Arc<Self>) -> JoinHandle<Result<Ws>> {
        let dial = Arc::clone(self);
        tokio::spawn(async move { dial.connect().await })
    }
}

/// Read until an event of type `want`; an `error` event or a close fails.
async fn expect(ws: &mut Ws, want: &str) -> Result<()> {
    loop {
        match ws.next().await {
            Some(Ok(Message::Text(text))) => {
                let msg: Value = serde_json::from_str(&text)?;
                match msg["type"].as_str() {
                    Some(t) if t == want => return Ok(()),
                    Some("error") => anyhow::bail!("voxtral realtime TTS: {}", error_message(&msg)),
                    _ => {}
                }
            }
            Some(Ok(Message::Close(frame))) => {
                anyhow::bail!("voxtral realtime TTS: closed before {want}: {frame:?}")
            }
            Some(Ok(_)) => {}
            Some(Err(e)) => return Err(e).context(format!("voxtral realtime TTS: waiting for {want}")),
            None => anyhow::bail!("voxtral realtime TTS: closed before {want}"),
        }
    }
}

fn error_message(msg: &Value) -> String {
    msg["error"]["message"]
        .as_str()
        .or(msg["message"].as_str())
        .unwrap_or("unknown error")
        .to_string()
}

fn append(text: &str) -> Message {
    Message::Text(json!({"type": "input_text.append", "text": text}).to_string().into())
}

fn flush() -> Message {
    Message::Text(json!({"type": "input_text.flush"}).to_string().into())
}

/// The next message on `active`, or never when there is none.
async fn read(active: &mut Option<Ws>) -> Option<Result<Message, tokio_tungstenite::tungstenite::Error>> {
    match active {
        Some(ws) => ws.next().await,
        None => std::future::pending().await,
    }
}

/// One session: owns the active connection and the spare.
struct Worker {
    dial: Arc<Dial>,
    active: Option<Ws>,
    spare: Option<JoinHandle<Result<Ws>>>,
    /// The utterance being written: every delta since the last flush, all of
    /// them sent on `active` unless `current_failed`.
    current: String,
    /// Why the utterance being written already failed; answered at its flush.
    current_failed: Option<String>,
    /// Flushed utterances sent on `active` and not answered yet, oldest first.
    /// An empty string is an empty flush, answered once it is at the front.
    pending: VecDeque<String>,
    /// Bytes of a sample cut across two deltas.
    remainder: Vec<u8>,
    events: mpsc::Sender<TtsEvent>,
    /// The event receiver was dropped.
    gone: bool,
    checked_format: bool,
}

impl Worker {
    async fn run(mut self, mut input: mpsc::Receiver<TtsInput>) {
        let mut input_open = true;
        while !self.gone {
            if !input_open && self.pending.is_empty() {
                break;
            }
            tokio::select! {
                biased;
                inp = input.recv(), if input_open => match inp {
                    Some(TtsInput::Text(t)) => self.on_text(t).await,
                    Some(TtsInput::Flush) => self.on_flush().await,
                    Some(TtsInput::Cancel) => self.on_cancel().await,
                    None => {
                        input_open = false;
                        if !self.current.trim().is_empty() || self.current_failed.is_some() {
                            self.on_flush().await;
                        }
                    }
                },
                msg = read(&mut self.active) => self.on_message(msg).await,
            }
        }
        if let Some(spare) = self.spare.take() {
            spare.abort();
        }
        if let Some(mut ws) = self.active.take() {
            let _ = ws.close(None).await;
        }
        debug!("voxtral realtime TTS: session closed");
    }

    async fn emit(&mut self, ev: TtsEvent) {
        if self.events.send(ev).await.is_err() {
            self.gone = true;
        }
    }

    /// A ready connection: the spare if it is still open, else a new one.
    /// A new spare is opened behind it.
    async fn take_conn(&mut self) -> Result<Ws> {
        let spare = match self.spare.take() {
            Some(handle) => match handle.await {
                Ok(Ok(mut ws)) => is_open(&mut ws).then_some(ws),
                Ok(Err(e)) => {
                    debug!(error = %e, "voxtral realtime TTS: spare failed to open");
                    None
                }
                _ => None,
            },
            None => None,
        };
        let ws = match spare {
            Some(ws) => ws,
            None => self.dial.connect().await?,
        };
        self.spare = Some(self.dial.spawn());
        Ok(ws)
    }

    /// Send `msg` on the active connection, opening one first if there is
    /// none (then the utterance being written is sent first, since its
    /// earlier deltas went to a connection that is gone).
    async fn send(&mut self, msg: Message) -> Result<()> {
        if self.active.is_none() {
            let mut ws = self.take_conn().await?;
            if !self.current.is_empty() {
                ws.send(append(&self.current)).await?;
            }
            self.active = Some(ws);
        }
        let ws = self.active.as_mut().expect("just opened");
        ws.send(msg).await.map_err(Into::into)
    }

    async fn on_text(&mut self, t: String) {
        if t.is_empty() || self.current_failed.is_some() {
            return;
        }
        // Without a connection `send` opens one and sends `current` before `t`.
        let res = self.send(append(&t)).await;
        self.current.push_str(&t);
        match res {
            Ok(()) => {}
            // No connection could be opened: this utterance fails at its flush.
            Err(e) if self.active.is_none() => self.current_failed = Some(format!("{e:#}")),
            Err(e) => self.lost(format!("{e:#}")).await,
        }
    }

    async fn on_flush(&mut self) {
        if let Some(e) = self.current_failed.take() {
            self.current.clear();
            self.emit(TtsEvent::Error(e)).await;
            return;
        }
        if self.current.trim().is_empty() {
            // Nothing to speak; whitespace already sent stays a harmless prefix.
            if self.pending.is_empty() {
                self.emit(TtsEvent::UtteranceDone).await;
            } else {
                self.pending.push_back(String::new());
            }
            return;
        }
        let res = self.send(flush()).await;
        self.pending.push_back(std::mem::take(&mut self.current));
        match res {
            Ok(()) => {}
            Err(e) if self.active.is_none() => self.fail_all(format!("{e:#}")).await,
            Err(e) => self.lost(format!("{e:#}")).await,
        }
    }

    async fn on_cancel(&mut self) {
        let busy = !self.pending.is_empty() || !self.current.is_empty();
        self.current.clear();
        self.current_failed = None;
        self.pending.clear();
        self.remainder.clear();
        if busy {
            if let Some(mut ws) = self.active.take() {
                tokio::spawn(async move {
                    let _ = ws.close(None).await;
                });
            }
            match self.take_conn().await {
                Ok(ws) => self.active = Some(ws),
                Err(e) => debug!(error = %e, "voxtral realtime TTS: no connection after cancel; retrying at the next text"),
            }
        }
        self.emit(TtsEvent::Cancelled).await;
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
                warn!(error = %e, "voxtral realtime TTS: unparsable event");
                return;
            }
        };
        match msg["type"].as_str().unwrap_or("") {
            "speech.audio.delta" => {
                let Some(b64) = msg["audio_data"].as_str() else { return };
                match STANDARD.decode(b64) {
                    Ok(bytes) => self.on_audio(bytes).await,
                    Err(e) => warn!(error = %e, "voxtral realtime TTS: bad base64 audio"),
                }
            }
            "speech.audio.done" => {
                self.remainder.clear();
                if self.pending.pop_front().is_some() {
                    self.emit(TtsEvent::UtteranceDone).await;
                    self.answer_empty_flushes().await;
                }
            }
            "error" => {
                let reason = error_message(&msg);
                warn!(error = %reason, "voxtral realtime TTS: generation failed");
                if let Some(mut ws) = self.active.take() {
                    tokio::spawn(async move {
                        let _ = ws.close(None).await;
                    });
                }
                self.lost(reason).await;
            }
            other => debug!(event = %other, "voxtral realtime TTS: ignored event"),
        }
    }

    async fn on_audio(&mut self, bytes: Vec<u8>) {
        self.remainder.extend_from_slice(&bytes);
        let whole = self.remainder.len() - self.remainder.len() % 4;
        if whole == 0 {
            return;
        }
        let data: Vec<u8> = self.remainder.drain(..whole).collect();
        if !self.checked_format {
            self.checked_format = true;
            let wild = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .filter(|s| !s.is_finite() || s.abs() > 4.0)
                .count();
            if wild * 10 > data.len() / 4 {
                warn!("voxtral realtime TTS: audio does not look like float32 PCM; check the format");
            }
        }
        self.emit(TtsEvent::Audio(AudioChunk::new(data))).await;
    }

    async fn answer_empty_flushes(&mut self) {
        while self.pending.front().is_some_and(String::is_empty) {
            self.pending.pop_front();
            self.emit(TtsEvent::UtteranceDone).await;
        }
    }

    /// The active connection is gone (closed, failed, or an `error` event):
    /// the generation in progress fails; the rest is sent again on a new
    /// connection.
    async fn lost(&mut self, reason: String) {
        self.active = None;
        self.remainder.clear();
        if self.pending.is_empty() && self.current.is_empty() {
            debug!(%reason, "voxtral realtime TTS: idle connection closed");
            return;
        }
        warn!(%reason, "voxtral realtime TTS: connection lost mid-generation");
        if self.pending.pop_front().is_some() {
            self.emit(TtsEvent::Error(reason)).await;
            self.answer_empty_flushes().await;
        } else if self.current_failed.is_none() {
            self.current_failed = Some(reason);
        }
        if self.pending.is_empty() && (self.current.is_empty() || self.current_failed.is_some()) {
            return;
        }
        // Send the queued utterances, then the one being written, again.
        let replay = async {
            let mut ws = self.take_conn().await?;
            for text in self.pending.iter().filter(|t| !t.is_empty()) {
                ws.send(append(text)).await?;
                ws.send(flush()).await?;
            }
            if !self.current.is_empty() && self.current_failed.is_none() {
                ws.send(append(&self.current)).await?;
            }
            anyhow::Ok(ws)
        };
        match replay.await {
            Ok(ws) => self.active = Some(ws),
            Err(e) => self.fail_all(format!("{e:#}")).await,
        }
    }

    /// No connection: every queued utterance fails, and the one being
    /// written fails at its flush.
    async fn fail_all(&mut self, reason: String) {
        self.active = None;
        while let Some(text) = self.pending.pop_front() {
            let ev = if text.is_empty() { TtsEvent::UtteranceDone } else { TtsEvent::Error(reason.clone()) };
            self.emit(ev).await;
        }
        if !self.current.is_empty() && self.current_failed.is_none() {
            self.current_failed = Some(reason);
        }
    }
}

/// Whether a connection that has sat unread is still open.
fn is_open(ws: &mut Ws) -> bool {
    match ws.next().now_or_never() {
        None => true,
        Some(Some(Ok(Message::Close(_)))) | Some(Some(Err(_))) | Some(None) => false,
        Some(Some(Ok(_))) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tokio::net::TcpListener;

    /// What the fake server does for one connection.
    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        /// Every append is answered at once by an audio delta of one f32
        /// sample per character, cut off sample boundaries; a flush by
        /// `speech.audio.done`.
        Echo,
        /// Like `Echo`, but the first flush is answered by an error event
        /// and a close.
        FailFirstFlush,
        /// Audio for appends, but a flush is never answered.
        Stall,
    }

    /// A fake Mistral realtime TTS server on 127.0.0.1. Connection `i` (in
    /// accept order) runs `modes[i]` (the last mode for the rest) and every
    /// message it received is logged as (`i`, message).
    struct FakeServer {
        url: String,
        log: Arc<Mutex<Vec<(usize, Value)>>>,
        accepted: Arc<Mutex<usize>>,
    }

    impl FakeServer {
        async fn start(modes: Vec<Mode>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let log = Arc::new(Mutex::new(Vec::new()));
            let accepted = Arc::new(Mutex::new(0));
            let (log2, accepted2) = (Arc::clone(&log), Arc::clone(&accepted));
            tokio::spawn(async move {
                loop {
                    let (tcp, _) = listener.accept().await.unwrap();
                    let i = {
                        let mut n = accepted2.lock().unwrap();
                        *n += 1;
                        *n - 1
                    };
                    let mode = *modes.get(i).or(modes.last()).unwrap();
                    tokio::spawn(serve(tcp, i, mode, Arc::clone(&log2)));
                }
            });
            Self { url, log, accepted }
        }

        fn types(&self, conn: usize) -> Vec<String> {
            self.log
                .lock()
                .unwrap()
                .iter()
                .filter(|(i, _)| *i == conn)
                .map(|(_, m)| match m["type"].as_str().unwrap() {
                    "input_text.append" => format!("append:{}", m["text"].as_str().unwrap()),
                    t => t.to_string(),
                })
                .collect()
        }
    }

    async fn serve(tcp: tokio::net::TcpStream, i: usize, mode: Mode, log: Arc<Mutex<Vec<(usize, Value)>>>) {
        let mut ws = tokio_tungstenite::accept_hdr_async(tcp, |req: &tokio_tungstenite::tungstenite::handshake::server::Request, resp| {
            assert!(req.uri().path().ends_with("/v1/audio/speech/realtime"));
            assert!(req.uri().query().unwrap().contains("model="));
            Ok(resp)
        })
        .await
        .unwrap();
        let text = |v: Value| Message::Text(v.to_string().into());
        ws.send(text(json!({"type": "session.created", "session": {}}))).await.unwrap();
        let mut flushed = 0;
        let mut configured = false;
        let mut carry: Vec<u8> = Vec::new();
        while let Some(Ok(msg)) = ws.next().await {
            let Message::Text(raw) = msg else { continue };
            let m: Value = serde_json::from_str(&raw).unwrap();
            log.lock().unwrap().push((i, m.clone()));
            match m["type"].as_str().unwrap() {
                "session.update" => {
                    assert!(!configured, "configured once");
                    assert_eq!(m["session"]["response_format"], "pcm");
                    configured = true;
                    ws.send(text(json!({"type": "session.updated", "session": m["session"]}))).await.unwrap();
                }
                "input_text.append" => {
                    assert!(configured, "text only after session.update");
                    let n = m["text"].as_str().unwrap().chars().count();
                    carry.extend((0..n).flat_map(|_| 0.25f32.to_le_bytes()));
                    // Send all but the last byte now: deltas off sample boundaries.
                    let keep = carry.split_off(carry.len() - 1);
                    let delta = std::mem::replace(&mut carry, keep);
                    ws.send(text(json!({"type": "speech.audio.delta", "audio_data": STANDARD.encode(delta)})))
                        .await
                        .unwrap();
                }
                "input_text.flush" => {
                    flushed += 1;
                    if mode == Mode::FailFirstFlush && flushed == 1 {
                        ws.send(text(json!({"type": "error", "error": {"message": "boom", "code": 500}})))
                            .await
                            .unwrap();
                        let _ = ws.close(None).await;
                        return;
                    }
                    if mode == Mode::Stall {
                        continue;
                    }
                    let rest = std::mem::take(&mut carry);
                    ws.send(text(json!({"type": "speech.audio.delta", "audio_data": STANDARD.encode(rest)})))
                        .await
                        .unwrap();
                    ws.send(text(json!({"type": "speech.audio.done", "usage": {}}))).await.unwrap();
                }
                _ => {}
            }
        }
    }

    async fn next(rx: &mut mpsc::Receiver<TtsEvent>) -> TtsEvent {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event in time")
            .expect("session open")
    }

    /// Audio bytes until the next non-audio event, which is returned.
    async fn audio_until(rx: &mut mpsc::Receiver<TtsEvent>) -> (usize, TtsEvent) {
        let mut bytes = 0;
        loop {
            match next(rx).await {
                TtsEvent::Audio(a) => {
                    assert_eq!(a.data.len() % 4, 0, "whole f32 samples");
                    bytes += a.data.len();
                }
                ev => return (bytes, ev),
            }
        }
    }

    async fn wait_for(cond: impl Fn() -> bool) {
        for _ in 0..200 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("condition never held");
    }

    #[tokio::test]
    async fn deltas_stream_audio_before_flush_and_each_flush_ends_an_utterance() {
        let server = FakeServer::start(vec![Mode::Echo]).await;
        let tts = VoxtralRealtimeTts::local(&server.url).with_model("voxtral-mini-tts-3-0");
        let (format, tx, mut rx) = tts.open(StreamingTtsConfig::new("en_paul_neutral")).await.unwrap();
        assert_eq!(format.encoding, AudioEncoding::F32);
        assert_eq!(format.sample_rate, 24_000);

        tx.send(TtsInput::Text("Hello ".into())).await.unwrap();
        let TtsEvent::Audio(first) = next(&mut rx).await else { panic!("audio before flush") };
        assert_eq!(first.data.len(), 20, "5 whole samples; the 6th is carried over");
        let s = f32::from_le_bytes([first.data[0], first.data[1], first.data[2], first.data[3]]);
        assert_eq!(s, 0.25);

        tx.send(TtsInput::Text("world.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let (bytes, ev) = audio_until(&mut rx).await;
        assert_eq!(20 + bytes, 4 * "Hello world.".len());
        assert!(matches!(ev, TtsEvent::UtteranceDone));

        tx.send(TtsInput::Text("Again.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap(); // empty: answered in order
        let (bytes, ev) = audio_until(&mut rx).await;
        assert_eq!(bytes, 4 * "Again.".len());
        assert!(matches!(ev, TtsEvent::UtteranceDone));
        assert!(matches!(next(&mut rx).await, TtsEvent::UtteranceDone));

        assert_eq!(
            server.types(0),
            vec![
                "session.update",
                "append:Hello ",
                "append:world.",
                "input_text.flush",
                "append:Again.",
                "input_text.flush",
            ]
        );
        let log = server.log.lock().unwrap();
        assert_eq!(log[0].1["session"]["voice"], "en_paul_neutral");
        assert_eq!(log[0].1["session"]["streaming_mode"], "responsive");
    }

    #[tokio::test]
    async fn cancel_closes_the_busy_connection_and_the_next_utterance_uses_the_spare() {
        let server = FakeServer::start(vec![Mode::Stall, Mode::Echo]).await;
        let tts = VoxtralRealtimeTts::local(&server.url);
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();
        wait_for(|| *server.accepted.lock().unwrap() == 2).await; // the spare is open

        tx.send(TtsInput::Text("A reply that is never finished.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        assert!(matches!(next(&mut rx).await, TtsEvent::Audio(_)));
        tx.send(TtsInput::Cancel).await.unwrap();
        let (_, ev) = audio_until(&mut rx).await;
        assert!(matches!(ev, TtsEvent::Cancelled));

        tx.send(TtsInput::Text("After.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let (bytes, ev) = audio_until(&mut rx).await;
        assert_eq!(bytes, 4 * "After.".len());
        assert!(matches!(ev, TtsEvent::UtteranceDone));

        assert_eq!(server.types(1), vec!["session.update", "append:After.", "input_text.flush"]);
        // A new spare replaced the one taken.
        wait_for(|| *server.accepted.lock().unwrap() == 3).await;
    }

    #[tokio::test]
    async fn a_failed_generation_reconnects_and_replays_the_queued_utterances() {
        let server = FakeServer::start(vec![Mode::FailFirstFlush, Mode::Echo]).await;
        let tts = VoxtralRealtimeTts::local(&server.url);
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();

        tx.send(TtsInput::Text("First.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        tx.send(TtsInput::Text("Second.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let (_, ev) = audio_until(&mut rx).await;
        let TtsEvent::Error(e) = ev else { panic!("error, got {ev:?}") };
        assert!(e.contains("boom"), "{e}");
        let (bytes, ev) = audio_until(&mut rx).await;
        assert!(matches!(ev, TtsEvent::UtteranceDone), "{ev:?}");
        assert!(bytes >= 4 * "Second.".len());

        // The next utterance works too.
        tx.send(TtsInput::Text("Third.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let (bytes, ev) = audio_until(&mut rx).await;
        assert_eq!(bytes, 4 * "Third.".len());
        assert!(matches!(ev, TtsEvent::UtteranceDone));
        let replayed = server.types(1);
        assert!(replayed.contains(&"append:Second.".to_string()), "{replayed:?}");
        assert!(!replayed.contains(&"append:First.".to_string()), "the failed one is not replayed");
    }

    #[tokio::test]
    async fn a_failure_with_nothing_else_queued_reconnects_on_the_next_utterance() {
        let server = FakeServer::start(vec![Mode::FailFirstFlush, Mode::Echo]).await;
        let tts = VoxtralRealtimeTts::local(&server.url);
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();

        tx.send(TtsInput::Text("Doomed.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let (_, ev) = audio_until(&mut rx).await;
        assert!(matches!(ev, TtsEvent::Error(_)), "{ev:?}");

        tx.send(TtsInput::Text("Fine.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let (bytes, ev) = audio_until(&mut rx).await;
        assert_eq!(bytes, 4 * "Fine.".len());
        assert!(matches!(ev, TtsEvent::UtteranceDone));
    }

    #[tokio::test]
    async fn open_fails_when_the_server_is_unreachable() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let tts = VoxtralRealtimeTts::local(url);
        assert!(tts.open(StreamingTtsConfig::default()).await.is_err());
    }

    #[test]
    fn the_url_follows_the_base_url_scheme() {
        let hosted = VoxtralRealtimeTts::new("k");
        assert_eq!(
            hosted.url(),
            "wss://api.mistral.ai/v1/audio/speech/realtime?model=voxtral-mini-tts-3-0"
        );
        let local = VoxtralRealtimeTts::local("http://localhost:8000/").with_model("m");
        assert_eq!(local.url(), "ws://localhost:8000/v1/audio/speech/realtime?model=m");
    }
}
