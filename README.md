# simply-voice

Speech-to-text, text-to-speech and voice activity detection behind three traits. Each provider is a Cargo feature.

The repo also ships `sv`, a small command line on top of the library: an easy way for a person or an agent to speak and listen.

## The shape

```
             ┌─ SttProvider ──── transcribe(chunk) · stream()
             ├─ TtsProvider ──── synthesize(text, voice) · stream() · voices()
simply-voice ├─ StreamingTts ─── open(cfg) → (format, text in, audio out)
             ├─ RealtimeProvider  connect(...)
             └─ VoiceActivityDetector  energy-based, emits speech start / end
```

* **Callers depend on the traits**, never on a provider.
* **Streaming is a channel pair**: send audio or text in, receive transcriptions or audio chunks out.

## Streaming speech out

`StreamingTts::open(StreamingTtsConfig::new(voice))` returns the session's `AudioFormat`, a `Sender<TtsInput>` and a `Receiver<TtsEvent>`.

* **In**: `TtsInput::Text(delta)` as text is written (spaces included), `Flush` at the end of each utterance, `Cancel` to drop everything queued or playing.
* **Out**: `TtsEvent::Audio(chunk)` in the declared format; one `UtteranceDone` or one `Error(why)` per `Flush`, in order; `Cancelled` per `Cancel`, after which nothing belongs to the dropped utterances.
* **The session stays usable** after a cancel or an error. Dropping the sender speaks what is queued, then closes the receiver. Keep reading events: a full event channel holds the session up, cancels included.
* **`VoxtralRealtimeTts`** (feature `voxtral`) streams for real over Mistral's realtime WebSocket: audio can start before the flush, and a cancel swaps in a spare connection opened in advance. Audio is float32 mono 24 kHz. `VoxtralProvider::realtime_tts()` makes one with the provider's key, server and model; `with_streaming_mode(VoxtralStreamingMode::LongForm)` trades latency for quality.
* **`BatchStreamingTts::new(provider, format)`** puts any `TtsProvider` behind the trait: text waits for the flush, then is synthesized sentence by sentence; a cancel drops the in-flight `synthesize` future. `format` is what the provider returns: Voxtral REST f32 at 24 kHz, pocket-tts f32 at `PocketTtsProvider::sample_rate()`, Gemini and ElevenLabs PCM16 at 24 kHz. The encoding may differ (it is converted); the sample rate may not.

## Streaming speech in

`StreamingStt::open(StreamingSttConfig::new(sample_rate))` returns a `Sender<SttInput>` and a `Receiver<SttEvent>`, meant to stay open while people may speak.

* **In**: `SttInput::Audio(chunk)`, PCM16 LE mono at the declared rate (16 kHz by default), and `EndOfTurn` when your own voice activity detector hears the end of a turn.
* **Out**, grouped into turns: `SpeechStarted`; `Partial(text)`, replacing the previous partial; `Final(text)`, appended as is, never revised; `EagerEnd` and `Resumed`; then exactly one `TurnEnded { reason }` per turn, after all its text. `Error(why)` doesn't end the turn, and the session goes on.
* **`caps().ends_turns`** says who ends the turn. A provider that does sends `SpeechStarted`, ends the turn itself and may ignore `EndOfTurn`, so keep sending audio, silence included. One that doesn't sends no `SpeechStarted` and answers every `EndOfTurn` with one `TurnEnded`, transcribing the audio sent since the previous one.
* **Dropping the sender** finishes what is in flight (an open turn ends with `TurnEnd::Closed`), then closes the receiver. Keep reading events.
* **`VoxtralRealtimeStt`** (feature `voxtral`) keeps one WebSocket to Mistral's realtime transcription open, with its turn detection:
  * Mistral's "Aggressive" values by default (a turn can end 300 ms into a pause, 2.5 s at most), set with `with_turn_detection(VoxtralTurnDetection)`.
  * A 240 ms streaming delay (`with_streaming_delay`).
  * After the provider's turn end, the turn stays open one streaming delay for its last words.
  * A dropped connection reconnects on its own. The audio sent meanwhile is kept, and an open turn ends with `TurnEnd::ConnectionLost`.
  * `VoxtralProvider::realtime_stt()` makes one with the provider's key, server and model.
* **`KyutaiSttProvider`** (feature `kyutai`) implements it too: its end-of-turn head ends the turns, and each word is a `Final`.
* **`BatchStreamingStt::new(provider)`** puts any `SttProvider` behind the trait, with `ends_turns: false`. Each `EndOfTurn` is transcribed while audio keeps arriving, so nothing said during a transcription is lost.
* **`VoiceActivityDetector::with_end_silence(duration)`** sets how much silence ends speech (500 ms by default). **`process_silence()`**, called on a timer, ends speech when no audio arrives at all.

## Providers

| feature | provider | STT | TTS | runs |
|---|---|---|---|---|
| `kyutai` | `KyutaiSttProvider` | ✅ | | locally, candle |
| `pocket-tts` | `PocketTtsProvider` | | ✅ | locally, candle |
| `whisper` | `WhisperProvider` | ✅ | | locally, whisper.cpp |
| `voxtral` | `VoxtralProvider`, `VoxtralRealtimeTts`, `VoxtralRealtimeStt` | ✅ | ✅ | remote API (Voxtral 3 models by default) |
| `gemini` | `GeminiProvider` | ✅ | ✅ | remote API |
| `elevenlabs` | `ElevenLabsProvider` | ✅ | ✅ | remote API |

* **`PocketTtsProvider::with_voice_dir`** adds custom voices: each `*.safetensors` embedding or `*.wav` clip in the directory is a voice named after its file stem.
* **`PocketTtsProvider` runs one synth at a time.** Dropping a `synthesize()` future (aborting its task) or a `stream()`'s audio receiver stops the synth within one frame and frees the model for the next.
* **`metal`** turns on Apple Silicon acceleration for the candle providers.
* **`whisper`** needs cmake to build.
* **No feature is on by default.**
* **`RealtimeProvider` is a trait only.** No provider in this crate implements it yet.

## Use

```toml
[dependencies]
simply-voice = { git = "https://github.com/simply-audio/simply-voice", features = ["kyutai", "pocket-tts", "metal"] }
```

* **Local models** download from Hugging Face on first use, into `~/.cache/huggingface`.

## sv

```
sv say "The build is green."     speak in the configured voice; returns when done
sv listen                        wait for one spoken turn, print it as one line of text
sv devices [--pick]              list audio devices, or choose and save them
```

* **Build**: `cargo build --release -p simply-voice-cli`, giving `target/release/sv`.
* **`sv listen` exits** 0 with the transcript on stdout, 1 when nobody started speaking within `--timeout` (30 s), 2 on an error.
* **Devices**: `--input` / `--output` on every command take a number from `sv devices` or part of a device name.
* **Config**: `~/.config/simply-voice/config.toml`, sections `[audio]`, `[stt]`, `[tts]`; unknown keys are ignored.
* **For agents**: the `voice-say` skill in `skills/voice-say` wraps `sv say` and `sv listen`.

More in [docs/guides/BUILD.md](docs/guides/BUILD.md).

## Tests

* **The Kyutai tests load the real model.** The first run downloads about 2 GB.
  * `cargo test -p simply-voice --features kyutai,metal --test kyutai -- --nocapture`
  * `--nocapture` prints the end-of-speech to final latency per utterance.
* **Fixtures** in `crates/simply-voice/tests/fixtures/` are 16 kHz mono PCM16, shared with the `sv` turn tests.
* `cargo test --workspace` runs everything else, the `sv` turn tests included (they load Kyutai too).

## Where it is used

* [lumina](https://github.com/poucet/lumina): the simply daemon and Aurora.
* `sv`, in this repo.

This crate started inside lumina. Its history came along.
