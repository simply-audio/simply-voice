//! Voice activity detection — energy-based state machine.
//!
//! Determines when the user is speaking by tracking audio energy levels.
//! Emits state transitions that drive the voice pipeline and client UI.

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// VAD state transitions.
#[derive(Debug, Clone)]
pub enum VadEvent {
    /// Speech started — user began talking.
    SpeechStart,
    /// Intermediate audio chunk during active speech.
    SpeechChunk(Vec<i16>),
    /// Speech ended — complete utterance ready for processing.
    SpeechEnd(Vec<i16>),
}

#[derive(Debug, Clone, PartialEq)]
enum State {
    Silence,
    PossibleSpeech,
    Speech,
    PossibleSilence,
}

#[derive(Clone)]
pub struct VoiceActivityDetector {
    energy_threshold: f32,
    silence_duration_ms: u64,
    speech_duration_ms: u64,
    /// Quiet this long ends a `PossibleSpeech` onset. Shorter dips (the gap
    /// between two words) keep it, so the first word is not thrown away.
    onset_dip_ms: u64,
    /// Audio kept from before an onset, so a soft start (an "h" or "s"
    /// below the threshold) reaches STT.
    pre_roll_ms: u64,
    state: State,
    state_start: Instant,
    /// Last chunk above the threshold, for `onset_dip_ms`.
    last_speech: Instant,
    accumulated: Vec<i16>,
    /// Recent chunks while silent, oldest first, each with its arrival time.
    pre_roll: VecDeque<(Instant, Vec<i16>)>,
}

impl VoiceActivityDetector {
    pub fn new() -> Self {
        Self {
            energy_threshold: 0.01,
            silence_duration_ms: 500,
            speech_duration_ms: 200,
            onset_dip_ms: 150,
            pre_roll_ms: 300,
            state: State::Silence,
            state_start: Instant::now(),
            last_speech: Instant::now(),
            accumulated: Vec::new(),
            pre_roll: VecDeque::new(),
        }
    }

    /// How long the speaker must be quiet before [`VadEvent::SpeechEnd`];
    /// 500 ms by default.
    pub fn with_end_silence(mut self, silence: Duration) -> Self {
        self.silence_duration_ms = silence.as_millis() as u64;
        self
    }

    /// Process a chunk of PCM16 samples. Returns a VAD event if a state transition occurred.
    pub fn process(&mut self, samples: &[i16]) -> Option<VadEvent> {
        self.process_at(samples, Instant::now())
    }

    /// Account for a time with no audio at all, as silence: call it on a
    /// timer when no chunk arrives. Speech ends once the end silence has
    /// passed since the last loud chunk, as if quiet chunks had been fed
    /// (they are not added to the utterance). Returns
    /// [`VadEvent::SpeechEnd`] when that happens.
    pub fn process_silence(&mut self) -> Option<VadEvent> {
        self.silence_at(Instant::now())
    }

    fn silence_at(&mut self, now: Instant) -> Option<VadEvent> {
        let quiet_for = now.duration_since(self.last_speech).as_millis();
        match self.state {
            State::Silence => None,
            State::PossibleSpeech => {
                if quiet_for >= self.onset_dip_ms as u128 {
                    debug!("VAD: PossibleSpeech -> Silence (no audio)");
                    self.transition(State::Silence, now);
                    let onset = std::mem::take(&mut self.accumulated);
                    self.push_pre_roll(now, onset);
                }
                None
            }
            State::Speech | State::PossibleSilence => {
                if self.state == State::Speech {
                    // The silence began after the last loud chunk.
                    self.transition(State::PossibleSilence, self.last_speech);
                }
                if now.duration_since(self.state_start).as_millis() >= self.silence_duration_ms as u128 {
                    Some(self.end_speech(now))
                } else {
                    None
                }
            }
        }
    }

    fn end_speech(&mut self, now: Instant) -> VadEvent {
        let audio = std::mem::take(&mut self.accumulated);
        self.transition(State::Silence, now);
        self.pre_roll.clear();
        info!("VAD: Speech ended ({} samples)", audio.len());
        VadEvent::SpeechEnd(audio)
    }

    fn process_at(&mut self, samples: &[i16], now: Instant) -> Option<VadEvent> {
        let energy = Self::energy(samples);
        let is_speech = energy > self.energy_threshold;
        let elapsed = now.duration_since(self.state_start);

        match self.state {
            State::Silence => {
                self.push_pre_roll(now, samples.to_vec());
                if is_speech {
                    debug!("VAD: Silence -> PossibleSpeech (energy: {energy:.4})");
                    self.transition(State::PossibleSpeech, now);
                    self.last_speech = now;
                    // The onset starts with the pre-roll, this chunk included.
                    self.accumulated = self.pre_roll.drain(..).flat_map(|(_, c)| c).collect();
                }
                None
            }
            State::PossibleSpeech => {
                self.accumulated.extend_from_slice(samples);
                if is_speech {
                    self.last_speech = now;
                }

                if is_speech && elapsed.as_millis() >= self.speech_duration_ms as u128 {
                    info!("VAD: PossibleSpeech -> Speech");
                    self.transition(State::Speech, now);
                    Some(VadEvent::SpeechStart)
                } else if !is_speech
                    && now.duration_since(self.last_speech).as_millis() >= self.onset_dip_ms as u128
                {
                    debug!("VAD: PossibleSpeech -> Silence (false positive)");
                    self.transition(State::Silence, now);
                    // Keep the abandoned onset as pre-roll: if speech resumes
                    // within `pre_roll_ms`, it is the start of the utterance.
                    let onset = std::mem::take(&mut self.accumulated);
                    self.push_pre_roll(now, onset);
                    None
                } else {
                    None
                }
            }
            State::Speech => {
                self.accumulated.extend_from_slice(samples);

                if is_speech {
                    self.last_speech = now;
                } else {
                    debug!("VAD: Speech -> PossibleSilence");
                    self.transition(State::PossibleSilence, now);
                }
                Some(VadEvent::SpeechChunk(samples.to_vec()))
            }
            State::PossibleSilence => {
                self.accumulated.extend_from_slice(samples);

                if is_speech {
                    debug!("VAD: PossibleSilence -> Speech (resumed)");
                    self.last_speech = now;
                    self.transition(State::Speech, now);
                    Some(VadEvent::SpeechChunk(samples.to_vec()))
                } else if elapsed.as_millis() >= self.silence_duration_ms as u128 {
                    Some(self.end_speech(now))
                } else {
                    None
                }
            }
        }
    }

    fn energy(samples: &[i16]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_squares: f64 = samples
            .iter()
            .map(|&s| {
                let f = s as f64 / 32768.0;
                f * f
            })
            .sum();
        (sum_squares / samples.len() as f64).sqrt() as f32
    }

    /// Append a chunk to the pre-roll and drop chunks older than `pre_roll_ms`.
    fn push_pre_roll(&mut self, now: Instant, chunk: Vec<i16>) {
        self.pre_roll.push_back((now, chunk));
        let horizon = Duration::from_millis(self.pre_roll_ms);
        while self
            .pre_roll
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > horizon)
        {
            self.pre_roll.pop_front();
        }
    }

    fn transition(&mut self, new_state: State, timestamp: Instant) {
        self.state = new_state;
        self.state_start = timestamp;
    }
}

impl Default for VoiceActivityDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOUD: [i16; 320] = [8_000; 320];
    const QUIET: [i16; 320] = [0; 320];

    fn ended(ev: Option<VadEvent>) -> bool {
        matches!(ev, Some(VadEvent::SpeechEnd(_)))
    }

    /// Speech for 300 ms in 20 ms chunks from `t0`; returns the time after.
    fn speak(vad: &mut VoiceActivityDetector, t0: Instant) -> Instant {
        let mut started = false;
        for i in 0..15 {
            let t = t0 + Duration::from_millis(20 * i);
            if let Some(VadEvent::SpeechStart) = vad.process_at(&LOUD, t) {
                started = true;
            }
        }
        assert!(started, "speech started");
        t0 + Duration::from_millis(300)
    }

    #[test]
    fn silence_without_audio_ends_speech_after_the_end_silence() {
        let t0 = Instant::now();
        let mut vad = VoiceActivityDetector::new().with_end_silence(Duration::from_millis(300));
        let last = speak(&mut vad, t0) - Duration::from_millis(20);
        assert!(vad.silence_at(last + Duration::from_millis(100)).is_none());
        assert!(vad.silence_at(last + Duration::from_millis(250)).is_none());
        let Some(VadEvent::SpeechEnd(audio)) = vad.silence_at(last + Duration::from_millis(300)) else {
            panic!("speech ended");
        };
        assert_eq!(audio.len(), 15 * 320, "no samples added for the gap");
        assert!(vad.silence_at(last + Duration::from_secs(2)).is_none(), "ends once");
    }

    #[test]
    fn quiet_chunks_and_missing_audio_count_toward_the_same_silence() {
        let t0 = Instant::now();
        let mut vad = VoiceActivityDetector::new();
        let after = speak(&mut vad, t0);
        // 200 ms of quiet chunks, then nothing: 500 ms after the last loud chunk.
        for i in 0..10 {
            assert!(!ended(vad.process_at(&QUIET, after + Duration::from_millis(20 * i))));
        }
        assert!(vad.silence_at(after + Duration::from_millis(400)).is_none());
        assert!(matches!(vad.silence_at(after + Duration::from_millis(500)), Some(VadEvent::SpeechEnd(_))));
    }

    #[test]
    fn the_default_end_silence_is_unchanged() {
        let t0 = Instant::now();
        let mut vad = VoiceActivityDetector::new();
        let after = speak(&mut vad, t0);
        assert!(!ended(vad.process_at(&QUIET, after)));
        assert!(!ended(vad.process_at(&QUIET, after + Duration::from_millis(480))));
        assert!(matches!(vad.process_at(&QUIET, after + Duration::from_millis(500)), Some(VadEvent::SpeechEnd(_))));
    }
}
