#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "voxtral")]
pub mod voxtral;

#[cfg(feature = "voxtral")]
pub mod voxtral_realtime_tts;

#[cfg(feature = "voxtral")]
pub mod voxtral_realtime_stt;

#[cfg(any(feature = "voxtral", feature = "gemini"))]
mod realtime_ws;

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "gemini")]
pub mod gemini_realtime_tts;

#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;

#[cfg(feature = "pocket-tts")]
pub mod pocket_tts;

#[cfg(feature = "kyutai")]
pub mod kyutai;

#[cfg(feature = "whisper")]
pub use whisper::WhisperProvider;

#[cfg(feature = "voxtral")]
pub use voxtral::VoxtralProvider;

#[cfg(feature = "voxtral")]
pub use voxtral_realtime_tts::{VoxtralRealtimeTts, VoxtralStreamingMode};

#[cfg(feature = "voxtral")]
pub use voxtral_realtime_stt::{VoxtralRealtimeStt, VoxtralTurnDetection};

#[cfg(feature = "gemini")]
pub use gemini::GeminiProvider;

#[cfg(feature = "gemini")]
pub use gemini_realtime_tts::GeminiRealtimeTts;

#[cfg(feature = "elevenlabs")]
pub use elevenlabs::ElevenLabsProvider;

#[cfg(feature = "pocket-tts")]
pub use pocket_tts::PocketTtsProvider;

#[cfg(feature = "kyutai")]
pub use kyutai::KyutaiSttProvider;
