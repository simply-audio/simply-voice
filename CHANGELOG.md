# Changelog

What got added, per landing — dated with a clock and the jj change id. The entry
format is stated once in AGENTS.md §2. Lanes write `changelog.d/<LANE>.md`;
it is folded in here on landing.

## Unreleased

#### Multi-channel audio routing in `sv` and `voice-say` · 2026-11-25 21:14 · vlulozlo

- `sv` now accepts `--output-channels` and `--input-channels` (1-based specs: `3`, `3,4`, `3-4`, `1,3-4`, or `all`) and the matching `[audio]` config keys `output_channels` and `input_channels` (array of 1-based integers or spec string).
- `sv say` routes mono speech onto the chosen output channels and leaves all other channels silent; `sv listen` mixes only the chosen input channels instead of averaging every channel on the device. Out-of-range channel specs fail immediately before loading the speech models.
- `sv devices` displays each device's channel count, and `sv devices --pick` prompts for channels when selecting a device with more than 2 channels. `skills/voice-say/SKILL.md` documents the new flags.

#### reqwest 0.13 for the HTTP providers · 2026-11-20 19:52 · klqmsuqs

- Voxtral, Gemini and ElevenLabs now use reqwest 0.13 and still connect over native-tls, as they did before. An app that already uses reqwest 0.13 with native-tls (Lumina, through rmcp) gets no new packages in its lock, and no longer has the dependency tree that made `cargo update -p simply-voice` fail its resolve. reqwest 0.12 is still in the tree through hf-hub (kyutai, pocket-tts).

#### Gemini behind the streaming traits · 2026-11-20 19:38 · kmkkmxyq

- **Gemini live transcription** (`GeminiRealtimeStt`, `GeminiProvider::realtime_stt()`): one Live API connection to `gemini-3.5-transcribe-live` per session, with interim text as `Partial` and a final per segment as `Final`. The server hears speech start, and an `EndOfTurn` from your own detector makes it finalize at once (hybrid detection). Connections are replaced between turns before Google's 10-minute limit, and a dropped connection keeps the open turn and sends its unfinalized audio again.
- **Gemini streaming speech** (`GeminiRealtimeTts`, `GeminiProvider::realtime_tts()`): `gemini-3.8-flash-lite-tts`, one streamed request per sentence. Speech starts at the first complete sentence, before the flush, and a cancel drops the request in flight.
- **`SentenceStreamingTts`** puts any provider that speaks one text per request with streamed audio (`SentenceTts`) behind `StreamingTts`. `SentenceChunker` cuts text into sentences as it arrives. `BatchStreamingTts` now runs on the same worker and behaves as before.
- The batch Gemini paths (`generateContent`) and their models are unchanged.

#### Streaming speech in, and Voxtral's realtime transcription with turn detection · 2026-11-20 18:44 · prznsspq

- New: `StreamingStt`, a transcription session kept open while people speak. Send audio, plus `EndOfTurn` when your own voice detector hears the end. You get back speech starts, text (`Final` is appended and never revised, `Partial` replaces itself), eager ends and resumes, and exactly one `TurnEnded { reason }` per turn. `caps().ends_turns` says whether the provider ends turns itself.
- New: `VoxtralRealtimeStt` (feature `voxtral`, also `VoxtralProvider::realtime_stt()`) keeps one WebSocket to Mistral's realtime transcription open, with Mistral's turn detection.
  - By default a turn can end 300 ms into a pause (Mistral's "Aggressive" values), and the text runs 240 ms behind the audio.
  - Each turn's last words still reach it after the provider's end of turn.
  - A dropped connection reconnects without the caller doing anything, and the audio sent meanwhile is kept.
- New: `BatchStreamingStt` puts any existing STT provider behind the same trait. Audio that arrives while a turn is being transcribed is kept for the next turn, so speech during a transcription is no longer dropped.
- New: `KyutaiSttProvider` implements `StreamingStt`, its end-of-turn detection ending the turns.
- New: `VoiceActivityDetector::with_end_silence(duration)` sets how much silence ends speech; the default stays 500 ms. `process_silence()` ends speech when no audio arrives at all.
- Changed: Voxtral speech-to-text now defaults to Voxtral Mini Transcribe Realtime 3 (`voxtral-mini-transcribe-realtime-3`), for the per-utterance `transcribe` too. `VoxtralProvider::local` keeps its own model.

#### Streaming speech out, and Voxtral's realtime TTS · 2026-11-20 18:24 · snsytkrx

- New: `StreamingTts`, speech that starts while the text is still being written. Open a session with a voice and get its audio format, a sender for text deltas (`Flush` ends an utterance, `Cancel` drops everything queued or playing) and a receiver for audio, one `UtteranceDone` or `Error` per utterance, and `Cancelled`. The session stays usable after a cancel or an error.
- New: `VoxtralRealtimeTts` (feature `voxtral`, also `VoxtralProvider::realtime_tts()`) speaks over Mistral's realtime WebSocket: audio can start before the utterance is complete, a cancel takes effect at once by swapping in a spare connection opened in advance, and a failed generation reconnects without the caller noticing. Audio is float32 mono 24 kHz.
- New: `BatchStreamingTts` puts any existing TTS provider (pocket-tts, Gemini, ElevenLabs, Voxtral REST) behind the same trait, synthesizing sentence by sentence after each flush; a cancel drops the synthesis in flight.
- Changed: Voxtral TTS now defaults to Voxtral TTS 3 (`voxtral-mini-tts-3-0`), for the REST path too. It accepts the old voices as well as its own. Speech-to-text is unchanged.

#### Gemini TTS audio behind a text part; voxtral builds on its own · 2026-11-20 18:22 · luwkloyn

- Gemini TTS no longer fails when its response puts a text part before the audio: it takes the first part that carries audio.
- The `voxtral` feature now builds without the caller's tokio turning on `macros` (the library turns it on itself).

#### Barge-in stops a local pocket-tts synth · 2026-11-20 18:18 · zoztopnk

- Dropping a `PocketTtsProvider::synthesize()` future (aborting the task that awaits it) now stops the synth within one frame and frees the model, so the next reply starts at once instead of waiting for the old one to finish. Dropping a `stream()`'s audio receiver does the same, mid-synth or while it waits for text. No API change.

#### Speak and listen from the command line (`sv`) · 2026-11-07 07:58 · rurmptyr

- New: `sv`, a small command line in this repo for people and agents. `sv say "…"` speaks in the configured voice (the Aurora voice by default) and returns when playback ends. `sv listen` waits for one spoken turn and prints it as one line of text. It exits 0 when it heard something, 1 when nobody spoke within `--timeout` (30 s by default), and 2 on an error. `sv devices [--pick]` lists the audio devices, or lets you choose them and saves the choice. `--input` / `--output` take a device number or part of its name.
- `sv` replaces voice-agent's `va` for saying, listening and picking devices. Its config is `~/.config/simply-voice/config.toml`. On the first run, if only `~/.config/voice-agent/config.toml` exists, it is copied across and the old file is left in place. Sections `sv` doesn't use are ignored.
- `sv listen --events` prints the old `va listen` timing stream (`--wav`, `--ptt`, `--tail`, `--backend`).
- The `voice-say` agent skill now lives here. It gained `scripts/listen`, so an agent can ask out loud and read the answer back.
- The library is unchanged for its users. It now sits at `crates/simply-voice` in a cargo workspace, and the git dependency `simply-voice = { git = "https://github.com/poucet/simply-voice" }` still finds it by name.
