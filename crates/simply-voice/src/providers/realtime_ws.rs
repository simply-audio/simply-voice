//! WebSocket plumbing shared by the realtime endpoints (Voxtral's
//! transcription and speech, Gemini's Live API): the URL, the connect (with a
//! Bearer key when there is one), and reading Voxtral's handshake events.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde_json::Value;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::{HeaderValue, AUTHORIZATION};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

pub(crate) type Ws = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// `base_url` (`https://…` → `wss://…`, `http://…` → `ws://…`) with
/// `path_and_query` appended.
pub(crate) fn ws_url(base_url: &str, path_and_query: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let ws = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        base.to_string()
    };
    format!("{ws}{path_and_query}")
}

/// Open a WebSocket to `url`, with `Authorization: Bearer <api_key>` unless
/// the key is empty (a local server).
pub(crate) async fn connect(url: &str, api_key: &str) -> Result<Ws> {
    let mut req = url.into_client_request()?;
    if !api_key.is_empty() {
        req.headers_mut()
            .insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {api_key}"))?);
    }
    let (ws, _) = connect_async(req).await?;
    Ok(ws)
}

/// Read until an event of type `want` and return it; an `error` event or a
/// close fails. `label` names the endpoint in errors.
pub(crate) async fn expect(ws: &mut Ws, want: &str, label: &str) -> Result<Value> {
    loop {
        match ws.next().await {
            Some(Ok(Message::Text(text))) => {
                let msg: Value = serde_json::from_str(&text)?;
                match msg["type"].as_str() {
                    Some(t) if t == want => return Ok(msg),
                    Some("error") => anyhow::bail!("{label}: {}", error_message(&msg)),
                    _ => {}
                }
            }
            Some(Ok(Message::Close(frame))) => anyhow::bail!("{label}: closed before {want}: {frame:?}"),
            Some(Ok(_)) => {}
            Some(Err(e)) => return Err(e).context(format!("{label}: waiting for {want}")),
            None => anyhow::bail!("{label}: closed before {want}"),
        }
    }
}

/// The message of an `error` event (`error.message`, or a top-level
/// `message`).
pub(crate) fn error_message(msg: &Value) -> String {
    msg["error"]["message"]
        .as_str()
        .or(msg["message"].as_str())
        .unwrap_or("unknown error")
        .to_string()
}
