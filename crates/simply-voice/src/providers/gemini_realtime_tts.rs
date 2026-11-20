//! Gemini streaming TTS: one streamed request per sentence through the
//! shared sentence worker ([`SentenceStreamingTts`]).
//!
//! Gemini's TTS models take a text per request and stream its audio back, but
//! not text as it is written. So text deltas are cut into sentences as they
//! complete and each sentence is one request, spoken in order; the first
//! sentence starts before the `Flush`. Prosody starts afresh at each sentence.
//!
//! Protocol (Google's *Text-to-speech generation* and *Streaming
//! interactions*): `POST {base}/interactions` with `x-goog-api-key`, body
//! `{model, input: [{type: "user_input", content: [{type: "text", text}]}],
//! response_format: {type: "audio", mime_type: "audio/l16", sample_rate:
//! 24000}, generation_config: {speech_config: [{voice}]}, stream: true}`. The
//! answer is server-sent events: `step.delta` events whose `delta` is
//! `{type: "audio", data: <base64>}` carry headerless PCM16 LE mono 24 kHz;
//! `interaction.completed` and `data: [DONE]` end it; an `error` event
//! (`{error: {message, code}}`) fails it.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use bytes::{Bytes, BytesMut};
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde_json::{json, Value};
use tracing::debug;

use super::gemini::{default_voice, DEFAULT_BASE_URL};
use crate::audio::AudioFormat;
use crate::provider::{AudioStream, SentenceStreamingTts, SentenceTts, StreamingTts, StreamingTtsConfig, TtsSession};

/// The TTS model for voice agents.
pub const DEFAULT_REALTIME_TTS_MODEL: &str = "gemini-3.8-flash-lite-tts";

/// What every chunk is: headerless PCM16 LE, mono, 24 kHz.
pub const GEMINI_REALTIME_TTS_FORMAT: AudioFormat = AudioFormat {
    encoding: crate::audio::AudioEncoding::Pcm16,
    sample_rate: 24_000,
    channels: 1,
};

/// Names this endpoint in errors.
const LABEL: &str = "gemini TTS";

/// Gemini text-to-speech behind [`StreamingTts`], one streamed request per
/// sentence. Audio is [`GEMINI_REALTIME_TTS_FORMAT`].
#[derive(Debug, Clone)]
pub struct GeminiRealtimeTts {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    style: Option<String>,
}

impl GeminiRealtimeTts {
    /// Google's hosted API, model `gemini-3.8-flash-lite-tts`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_REALTIME_TTS_MODEL.to_string(),
            style: None,
        }
    }

    /// The API root, up to and including the version
    /// (`https://generativelanguage.googleapis.com/v1beta` by default).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// A delivery style for every sentence (`speech_metadata.style`, e.g.
    /// "casual, friendly"); none by default. Google advises one short,
    /// constant style.
    pub fn with_style(mut self, style: impl Into<String>) -> Self {
        self.style = Some(style.into());
        self
    }

    fn body(&self, text: &str, voice: &str) -> Value {
        let mut item = json!({"type": "text", "text": text});
        if let Some(style) = &self.style {
            item["annotations"] = json!([{"type": "speech_metadata", "style": style}]);
        }
        let voice = if voice.is_empty() { default_voice() } else { voice };
        json!({
            "model": self.model,
            "input": [{"type": "user_input", "content": [item]}],
            "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": GEMINI_REALTIME_TTS_FORMAT.sample_rate},
            "generation_config": {"speech_config": [{"voice": voice}]},
            "stream": true,
        })
    }

    /// Send the request; the response, once its status is a success.
    async fn request(&self, text: &str, voice: &str) -> Result<reqwest::Response> {
        let url = format!("{}/interactions", self.base_url.trim_end_matches('/'));
        debug!(model = %self.model, chars = text.len(), "gemini TTS: speaking a sentence");
        let resp = self
            .client
            .post(url)
            .header("x-goog-api-key", &self.api_key)
            .json(&self.body(text, voice))
            .send()
            .await
            .context("gemini TTS: request")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
                .unwrap_or(body);
            anyhow::bail!("{LABEL} ({status}): {msg}");
        }
        Ok(resp)
    }
}

#[async_trait]
impl StreamingTts for GeminiRealtimeTts {
    async fn open(&self, cfg: StreamingTtsConfig) -> Result<TtsSession> {
        SentenceStreamingTts::new(Arc::new(self.clone())).open(cfg).await
    }
}

/// Where one request's audio stream is.
enum Sse {
    /// The request has not been sent yet.
    Start(GeminiRealtimeTts, String, String),
    /// Reading the event stream.
    Reading(Box<SseReader>),
    Done,
}

impl SentenceTts for GeminiRealtimeTts {
    fn format(&self) -> AudioFormat {
        GEMINI_REALTIME_TTS_FORMAT
    }

    fn speak(&self, text: String, voice: String) -> AudioStream {
        stream::unfold(Sse::Start(self.clone(), text, voice), |state| async move {
            let mut reader = match state {
                Sse::Done => return None,
                Sse::Start(tts, text, voice) => match tts.request(&text, &voice).await {
                    Ok(resp) => Box::new(SseReader::new(resp)),
                    Err(e) => return Some((Err(e), Sse::Done)),
                },
                Sse::Reading(reader) => reader,
            };
            match reader.next_audio().await {
                Ok(Some(audio)) => Some((Ok(audio), Sse::Reading(reader))),
                Ok(None) => None,
                Err(e) => Some((Err(e), Sse::Done)),
            }
        })
        .boxed()
    }
}

/// Reads audio out of a server-sent event stream, whole samples at a time.
struct SseReader {
    resp: reqwest::Response,
    buf: BytesMut,
    /// A byte of a sample whose other byte is in the next delta.
    odd: Option<u8>,
}

impl SseReader {
    fn new(resp: reqwest::Response) -> Self {
        Self { resp, buf: BytesMut::new(), odd: None }
    }

    /// The next audio chunk, or `None` at the end of the stream.
    async fn next_audio(&mut self) -> Result<Option<Bytes>> {
        loop {
            while let Some(event) = take_event(&mut self.buf) {
                match parse_event(&event)? {
                    Event::Audio(pcm) => {
                        let mut pcm = pcm;
                        if let Some(b) = self.odd.take() {
                            pcm.insert(0, b);
                        }
                        if pcm.len() % 2 == 1 {
                            self.odd = pcm.pop();
                        }
                        if !pcm.is_empty() {
                            return Ok(Some(pcm.into()));
                        }
                    }
                    Event::End => return Ok(None),
                    Event::Other => {}
                }
            }
            match self.resp.chunk().await.context("gemini TTS: reading the stream")? {
                Some(chunk) => self.buf.extend_from_slice(&chunk),
                // The server closed without `[DONE]`: whatever came is all.
                None => return Ok(None),
            }
        }
    }
}

/// One whole event's text out of `buf` (up to a blank line), if there is one.
fn take_event(buf: &mut BytesMut) -> Option<String> {
    let text = std::str::from_utf8(buf).ok()?;
    let (end, sep) = [("\r\n\r\n", 4), ("\n\n", 2)]
        .iter()
        .filter_map(|(s, n)| text.find(s).map(|i| (i, *n)))
        .min()?;
    let event = text[..end].to_string();
    let _ = buf.split_to(end + sep);
    Some(event)
}

enum Event {
    Audio(Vec<u8>),
    End,
    Other,
}

/// The meaning of one server-sent event (`event:` and `data:` lines).
fn parse_event(event: &str) -> Result<Event> {
    let data: Vec<&str> = event
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(|d| d.strip_prefix(' ').unwrap_or(d))
        .collect();
    if data.is_empty() {
        return Ok(Event::Other);
    }
    let data = data.join("\n");
    if data.trim() == "[DONE]" {
        return Ok(Event::End);
    }
    let v: Value = serde_json::from_str(&data).with_context(|| format!("gemini TTS: unparsable event {data:?}"))?;
    match v["event_type"].as_str().unwrap_or("") {
        "step.delta" if v["delta"]["type"] == "audio" => {
            let b64 = v["delta"]["data"].as_str().unwrap_or("");
            Ok(Event::Audio(STANDARD.decode(b64).context("gemini TTS: audio is not base64")?))
        }
        "error" => {
            let msg = v["error"]["message"].as_str().unwrap_or("unknown error");
            anyhow::bail!("{LABEL}: {msg}")
        }
        "interaction.completed" => match v["interaction"]["status"].as_str() {
            Some("failed") | Some("cancelled") => anyhow::bail!("{LABEL}: the interaction ended {}", v["interaction"]["status"]),
            _ => Ok(Event::End),
        },
        _ => Ok(Event::Other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{TtsEvent, TtsInput};
    use std::sync::Mutex;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::mpsc;

    /// A fake Gemini interactions endpoint on 127.0.0.1. Each request's JSON
    /// body is logged; the answer streams, per sentence, two audio deltas of
    /// one PCM16 sample per character (the second split at an odd byte, as a
    /// real delta may be), 30 ms apart. A text containing "stall" gets its
    /// first delta and then nothing, and sets `closed` when the client hangs
    /// up; one containing "fail" gets an `error` event; "http" a 400.
    struct FakeTts {
        url: String,
        bodies: Arc<Mutex<Vec<Value>>>,
        closed: Arc<Mutex<bool>>,
    }

    impl FakeTts {
        async fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/v1beta", listener.local_addr().unwrap());
            let bodies = Arc::new(Mutex::new(Vec::new()));
            let closed = Arc::new(Mutex::new(false));
            let (b, c) = (Arc::clone(&bodies), Arc::clone(&closed));
            tokio::spawn(async move {
                loop {
                    let (tcp, _) = listener.accept().await.unwrap();
                    tokio::spawn(serve(tcp, Arc::clone(&b), Arc::clone(&c)));
                }
            });
            Self { url, bodies, closed }
        }

        fn texts(&self) -> Vec<String> {
            self.bodies
                .lock()
                .unwrap()
                .iter()
                .map(|b| b["input"][0]["content"][0]["text"].as_str().unwrap().to_string())
                .collect()
        }
    }

    async fn serve(mut tcp: TcpStream, bodies: Arc<Mutex<Vec<Value>>>, closed: Arc<Mutex<bool>>) {
        let mut raw = Vec::new();
        let mut buf = [0u8; 4096];
        let (head_end, len) = loop {
            let n = tcp.read(&mut buf).await.unwrap();
            if n == 0 {
                return;
            }
            raw.extend_from_slice(&buf[..n]);
            if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&raw[..i]).to_lowercase();
                assert!(head.starts_with("post /v1beta/interactions "), "{head}");
                assert!(head.contains("x-goog-api-key: key"), "{head}");
                let len = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .map(|v| v.trim().parse::<usize>().unwrap())
                    .unwrap();
                break (i + 4, len);
            }
        };
        while raw.len() < head_end + len {
            let n = tcp.read(&mut buf).await.unwrap();
            raw.extend_from_slice(&buf[..n]);
        }
        let body: Value = serde_json::from_slice(&raw[head_end..head_end + len]).unwrap();
        let text = body["input"][0]["content"][0]["text"].as_str().unwrap().to_string();
        bodies.lock().unwrap().push(body);

        if text.contains("http") {
            let err = r#"{"error":{"code":400,"message":"bad voice"}}"#;
            let resp = format!("HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{err}", err.len());
            let _ = tcp.write_all(resp.as_bytes()).await;
            return;
        }
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
        tcp.write_all(head.as_bytes()).await.unwrap();
        let event = |v: Value| format!("event: {}\ndata: {v}\n\n", v["event_type"].as_str().unwrap());
        let pcm: Vec<u8> = text.chars().flat_map(|_| 7i16.to_le_bytes()).collect();
        let audio = |b: &[u8]| event(json!({"index": 0, "event_type": "step.delta", "delta": {"type": "audio", "data": STANDARD.encode(b)}}));
        let mut out = event(json!({"event_type": "interaction.created", "interaction": {"id": "v1_x", "status": "in_progress"}}));
        out += &event(json!({"index": 0, "event_type": "step.start", "step": {"type": "model_output"}}));
        if text.contains("fail") {
            out += &event(json!({"event_type": "error", "error": {"message": "synthesis failed", "code": "internal"}}));
            let _ = tcp.write_all(out.as_bytes()).await;
            return;
        }
        let half = pcm.len() / 2 + 1; // an odd split
        out += &audio(&pcm[..half]);
        tcp.write_all(out.as_bytes()).await.unwrap();
        if text.contains("stall") {
            // Wait for the client to hang up.
            let n = tcp.read(&mut buf).await.unwrap_or(0);
            assert_eq!(n, 0);
            *closed.lock().unwrap() = true;
            return;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
        let mut rest = audio(&pcm[half..]);
        rest += &event(json!({"index": 0, "event_type": "step.stop"}));
        rest += &event(json!({"event_type": "interaction.completed", "interaction": {"id": "v1_x", "status": "completed"}}));
        rest += "event: done\ndata: [DONE]\n\n";
        let _ = tcp.write_all(rest.as_bytes()).await;
    }

    async fn next(rx: &mut mpsc::Receiver<TtsEvent>) -> TtsEvent {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event in time")
            .expect("session open")
    }

    /// Audio bytes until the next non-audio event, which is returned too.
    async fn audio_until_answer(rx: &mut mpsc::Receiver<TtsEvent>) -> (usize, TtsEvent) {
        let mut n = 0;
        loop {
            match next(rx).await {
                TtsEvent::Audio(a) => {
                    assert_eq!(a.data.len() % 2, 0, "whole samples");
                    n += a.data.len();
                }
                other => return (n, other),
            }
        }
    }

    #[tokio::test]
    async fn sentences_are_requested_in_order_and_stream_as_they_come() {
        let server = FakeTts::start().await;
        let tts = GeminiRealtimeTts::new("key").with_base_url(&server.url).with_style("casual");
        let (format, tx, mut rx) = tts.open(StreamingTtsConfig::new("Kore")).await.unwrap();
        assert_eq!(format.sample_rate, 24_000);

        let first = "The first sentence is long enough.";
        let second = "Then a second sentence follows it.";
        tx.send(TtsInput::Text(format!("{first} "))).await.unwrap();
        // Spoken before the flush.
        let TtsEvent::Audio(a0) = next(&mut rx).await else { panic!("audio") };
        tx.send(TtsInput::Text(format!("{second} and a trailing fragment"))).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let (bytes, answer) = audio_until_answer(&mut rx).await;
        assert!(matches!(answer, TtsEvent::UtteranceDone), "{answer:?}");
        let total = 2 * (first.len() + second.len() + "and a trailing fragment".len());
        assert_eq!(a0.data.len() + bytes, total, "every sample arrives, split deltas included");
        assert_eq!(server.texts(), vec![first, second, "and a trailing fragment"]);

        let body = &server.bodies.lock().unwrap()[0];
        assert_eq!(body["model"], DEFAULT_REALTIME_TTS_MODEL);
        assert_eq!(body["stream"], true);
        assert_eq!(body["generation_config"]["speech_config"][0]["voice"], "Kore");
        assert_eq!(body["response_format"], json!({"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000}));
        assert_eq!(body["input"][0]["content"][0]["annotations"][0]["style"], "casual");
    }

    #[tokio::test]
    async fn cancel_mid_sentence_hangs_up_and_skips_the_queue() {
        let server = FakeTts::start().await;
        let tts = GeminiRealtimeTts::new("key").with_base_url(&server.url);
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();
        tx.send(TtsInput::Text("This sentence will stall in the middle. Never requested at all.".into()))
            .await
            .unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        assert!(matches!(next(&mut rx).await, TtsEvent::Audio(_)));
        tx.send(TtsInput::Cancel).await.unwrap();
        assert!(matches!(next(&mut rx).await, TtsEvent::Cancelled));
        for _ in 0..200 {
            if *server.closed.lock().unwrap() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(*server.closed.lock().unwrap(), "the stalled request was dropped");

        tx.send(TtsInput::Text("After it.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let (bytes, answer) = audio_until_answer(&mut rx).await;
        assert!(matches!(answer, TtsEvent::UtteranceDone));
        assert_eq!(bytes, 2 * "After it.".len());
        assert_eq!(server.texts().len(), 2, "the queued sentence was never requested");
        // The default voice is sent when the config names none.
        assert_eq!(server.bodies.lock().unwrap()[1]["generation_config"]["speech_config"][0]["voice"], default_voice());
    }

    #[tokio::test]
    async fn a_flush_sends_a_trailing_fragment_alone() {
        let server = FakeTts::start().await;
        let tts = GeminiRealtimeTts::new("key").with_base_url(&server.url);
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();
        tx.send(TtsInput::Text("Short".into())).await.unwrap();
        tx.send(TtsInput::Text(" bit".into())).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(server.texts().is_empty(), "a fragment waits for the flush");
        tx.send(TtsInput::Flush).await.unwrap();
        let (bytes, answer) = audio_until_answer(&mut rx).await;
        assert!(matches!(answer, TtsEvent::UtteranceDone));
        assert_eq!(bytes, 2 * "Short bit".len());
        assert_eq!(server.texts(), vec!["Short bit"]);
    }

    #[tokio::test]
    async fn server_errors_fail_the_utterance_and_the_session_goes_on() {
        let server = FakeTts::start().await;
        let tts = GeminiRealtimeTts::new("key").with_base_url(&server.url);
        let (_, tx, mut rx) = tts.open(StreamingTtsConfig::default()).await.unwrap();
        tx.send(TtsInput::Text("Please fail this one.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let TtsEvent::Error(e) = next(&mut rx).await else { panic!("error") };
        assert!(e.contains("synthesis failed"), "{e}");
        tx.send(TtsInput::Text("An http error now.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let TtsEvent::Error(e) = next(&mut rx).await else { panic!("error") };
        assert!(e.contains("400") && e.contains("bad voice"), "{e}");
        tx.send(TtsInput::Text("Fine.".into())).await.unwrap();
        tx.send(TtsInput::Flush).await.unwrap();
        let (bytes, answer) = audio_until_answer(&mut rx).await;
        assert!(matches!(answer, TtsEvent::UtteranceDone));
        assert_eq!(bytes, 10);
    }

    #[test]
    fn events_split_across_reads_and_line_endings() {
        let mut buf = BytesMut::from(&b"event: step.delta\r\ndata: {\"a\":1}\r\n\r\ndata: [DO"[..]);
        assert_eq!(take_event(&mut buf).as_deref(), Some("event: step.delta\r\ndata: {\"a\":1}"));
        assert_eq!(take_event(&mut buf), None);
        buf.extend_from_slice(b"NE]\n\n");
        assert!(matches!(parse_event(&take_event(&mut buf).unwrap()).unwrap(), Event::End));
    }
}
