# Changelog

What got added, per landing — dated with a clock and the jj change id. The entry
format is stated once in AGENTS.md §2. Lanes write `changelog.d/<LANE>.md`;
it is folded in here on landing.

## Unreleased

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
