//! VAD operator.
//!
//! Consumes per-speaker [`AudioChunk`]s, runs Silero VAD, emits
//! closed [`VoiceRegion`]s. Streaming-capable: LSTM state is held
//! per pseudo_id across chunks; a region only closes when enough
//! trailing silence accumulates.
//!
//! The actual inference is isolated behind the [`VadEngine`] trait so
//! tests can feed synthetic frame probabilities without loading ONNX.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::config::VadConfig;
use crate::error::PipelineError;
use crate::operator::Operator;
use crate::types::{AudioChunk, PseudoId, SessionId, VoiceRegion};

// ---------------------------------------------------------------------------
// VAD engine trait
// ---------------------------------------------------------------------------

/// The numeric inner loop of VAD. Takes f32 mono 16 kHz PCM, returns
/// per-frame speech probabilities. LSTM state is owned internally per
/// [`VadContext`].
pub trait VadEngine: Send + 'static {
    /// Open a new speaker context.
    fn new_context(&self) -> Box<dyn VadContext>;
}

pub trait VadContext: Send {
    /// Feed mono 16 kHz f32 samples; return per-frame probabilities.
    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, PipelineError>;
}

/// 576-sample frames at 16 kHz = 36 ms per frame. Shared between the
/// engine and the tracker.
pub const FRAME_SIZE: usize = 576;
pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_DURATION_MS: f32 = (FRAME_SIZE as f32) * 1000.0 / (SAMPLE_RATE as f32);

// ---------------------------------------------------------------------------
// Per-speaker region tracker (engine-agnostic)
// ---------------------------------------------------------------------------

/// Tracks open/closed regions for one speaker using probabilities fed
/// from the engine. Pure state machine, no I/O.
struct RegionTracker {
    threshold: f32,
    min_speech_frames: u32,
    min_silence_frames: u32,
    pad_frames: u32,

    session_id: SessionId,
    pseudo_id: PseudoId,

    /// 16 kHz f32 PCM accumulated so far for this speaker. Used to
    /// extract the audio slice for a closed region.
    pcm: Vec<f32>,
    frames_emitted: u32,

    state: TrackerState,
}

#[derive(Debug)]
enum TrackerState {
    Silence,
    PotentialSpeech { start_frame: u32, consec_speech: u32 },
    Speech { start_frame: u32 },
    PotentialSilence { start_frame: u32, consec_silence: u32 },
}

impl RegionTracker {
    fn new(config: &VadConfig, session_id: SessionId, pseudo_id: PseudoId) -> Self {
        let min_speech_frames =
            ((config.min_speech_ms as f32) / FRAME_DURATION_MS).ceil() as u32;
        let min_silence_frames =
            ((config.min_silence_ms as f32) / FRAME_DURATION_MS).ceil() as u32;
        let pad_frames = ((config.pad_ms as f32) / FRAME_DURATION_MS).round() as u32;
        Self {
            threshold: config.threshold,
            min_speech_frames,
            min_silence_frames,
            pad_frames,
            session_id,
            pseudo_id,
            pcm: Vec::new(),
            frames_emitted: 0,
            state: TrackerState::Silence,
        }
    }

    fn extend_pcm(&mut self, samples: &[f32]) {
        self.pcm.extend_from_slice(samples);
    }

    /// Step one frame. Returns a region if this frame closes one.
    fn step(&mut self, frame_idx: u32, prob: f32) -> Option<VoiceRegion> {
        let is_speech = prob >= self.threshold;
        match self.state {
            TrackerState::Silence if is_speech => {
                self.state = TrackerState::PotentialSpeech {
                    start_frame: frame_idx,
                    consec_speech: 1,
                };
                None
            }
            TrackerState::Silence => None,
            TrackerState::PotentialSpeech {
                start_frame,
                consec_speech,
            } => {
                if is_speech {
                    let consec_speech = consec_speech + 1;
                    if consec_speech >= self.min_speech_frames {
                        self.state = TrackerState::Speech { start_frame };
                    } else {
                        self.state = TrackerState::PotentialSpeech {
                            start_frame,
                            consec_speech,
                        };
                    }
                } else {
                    // Speech wobble didn't stick. Drop back to silence.
                    self.state = TrackerState::Silence;
                }
                None
            }
            TrackerState::Speech { start_frame } => {
                if is_speech {
                    None
                } else {
                    self.state = TrackerState::PotentialSilence {
                        start_frame,
                        consec_silence: 1,
                    };
                    None
                }
            }
            TrackerState::PotentialSilence {
                start_frame,
                consec_silence,
            } => {
                if is_speech {
                    // False alarm — still speaking.
                    self.state = TrackerState::Speech { start_frame };
                    None
                } else {
                    let consec_silence = consec_silence + 1;
                    if consec_silence >= self.min_silence_frames {
                        // Close the region. `end_frame` = last speech frame.
                        let end_frame = frame_idx + 1 - consec_silence;
                        self.state = TrackerState::Silence;
                        Some(self.make_region(start_frame, end_frame))
                    } else {
                        self.state = TrackerState::PotentialSilence {
                            start_frame,
                            consec_silence,
                        };
                        None
                    }
                }
            }
        }
    }

    /// Flush any open speech region at end-of-input.
    fn finalize(&mut self, total_frames: u32) -> Option<VoiceRegion> {
        let state = std::mem::replace(&mut self.state, TrackerState::Silence);
        match state {
            TrackerState::Speech { start_frame }
            | TrackerState::PotentialSilence { start_frame, .. } => {
                Some(self.make_region(start_frame, total_frames))
            }
            _ => None,
        }
    }

    fn make_region(&self, start_frame: u32, end_frame: u32) -> VoiceRegion {
        // Pad both ends (clamped to valid range).
        let padded_start = start_frame.saturating_sub(self.pad_frames);
        let padded_end = (end_frame + self.pad_frames).min(self.total_frames_seen());

        let start_ms = frame_to_ms(padded_start);
        let end_ms = frame_to_ms(padded_end);

        let start_sample = padded_start as usize * FRAME_SIZE;
        let end_sample = (padded_end as usize * FRAME_SIZE).min(self.pcm.len());
        let pcm = if start_sample < end_sample {
            self.pcm[start_sample..end_sample].to_vec()
        } else {
            Vec::new()
        };

        VoiceRegion {
            session_id: self.session_id,
            pseudo_id: self.pseudo_id.clone(),
            start_ms,
            end_ms,
            pcm: Arc::from(pcm),
        }
    }

    fn total_frames_seen(&self) -> u32 {
        (self.pcm.len() / FRAME_SIZE) as u32
    }
}

fn frame_to_ms(frame: u32) -> u64 {
    (frame as u64 * FRAME_SIZE as u64 * 1000) / SAMPLE_RATE as u64
}

// ---------------------------------------------------------------------------
// Operator
// ---------------------------------------------------------------------------

pub struct VadOperator {
    config: VadConfig,
    engine: Box<dyn VadEngine>,
    speakers: HashMap<PseudoId, SpeakerState>,
    pending: Vec<VoiceRegion>,
}

struct SpeakerState {
    tracker: RegionTracker,
    ctx: Box<dyn VadContext>,
    /// Leftover samples that did not fill a frame in the last call.
    residual: Vec<f32>,
}

impl VadOperator {
    pub fn new(config: VadConfig, engine: Box<dyn VadEngine>) -> Self {
        Self {
            config,
            engine,
            speakers: HashMap::new(),
            pending: Vec::new(),
        }
    }

    fn speaker_state_mut(
        &mut self,
        session_id: SessionId,
        pseudo_id: &PseudoId,
    ) -> &mut SpeakerState {
        if !self.speakers.contains_key(pseudo_id) {
            let tracker = RegionTracker::new(&self.config, session_id, pseudo_id.clone());
            let ctx = self.engine.new_context();
            self.speakers.insert(
                pseudo_id.clone(),
                SpeakerState {
                    tracker,
                    ctx,
                    residual: Vec::new(),
                },
            );
        }
        self.speakers
            .get_mut(pseudo_id)
            .expect("speaker state just inserted")
    }
}

#[async_trait]
impl Operator for VadOperator {
    type Input = AudioChunk;
    type Output = VoiceRegion;

    async fn ingest(&mut self, input: AudioChunk) -> Result<(), PipelineError> {
        let span = tracing::info_span!("operator", name = "vad");
        let _enter = span.enter();

        if input.pcm.is_empty() {
            return Ok(());
        }

        // Downsample 48 kHz → 16 kHz via decimation-with-averaging (3:1).
        let mono_16k = downsample_48k_to_16k_i16(&input.pcm);

        let regions = {
            let state = self.speaker_state_mut(input.session_id, &input.pseudo_id);

            state.tracker.extend_pcm(&mono_16k);

            let mut combined = std::mem::take(&mut state.residual);
            combined.extend_from_slice(&mono_16k);
            let frame_count = combined.len() / FRAME_SIZE;
            let process_len = frame_count * FRAME_SIZE;
            let residual = combined.split_off(process_len);
            state.residual = residual;

            if frame_count == 0 {
                Vec::new()
            } else {
                let probs = state
                    .ctx
                    .process(&combined[..process_len])
                    .map_err(|e| PipelineError::Vad(format!("engine: {e}")))?;

                if probs.len() != frame_count {
                    return Err(PipelineError::Vad(format!(
                        "engine returned {} probs for {} frames",
                        probs.len(),
                        frame_count
                    )));
                }

                let frames_seen_before = state.tracker.frames_emitted;
                let mut regions = Vec::new();
                for (i, prob) in probs.into_iter().enumerate() {
                    let frame_idx = frames_seen_before + i as u32;
                    if let Some(region) = state.tracker.step(frame_idx, prob) {
                        regions.push(region);
                    }
                }
                state.tracker.frames_emitted += frame_count as u32;
                regions
            }
        };

        for region in regions {
            metrics::counter!(
                "chronicle_pipeline_operator_emit_count",
                "operator" => "vad",
            )
            .increment(1);
            self.pending.push(region);
        }

        Ok(())
    }

    fn emit(&mut self) -> Vec<VoiceRegion> {
        std::mem::take(&mut self.pending)
    }

    async fn finalize(&mut self) -> Result<Vec<VoiceRegion>, PipelineError> {
        let span = tracing::info_span!("operator", name = "vad");
        let _enter = span.enter();

        // Flush residual through the engine by zero-padding to a frame.
        let speaker_ids: Vec<PseudoId> = self.speakers.keys().cloned().collect();
        let mut all_regions = Vec::new();
        for pseudo_id in speaker_ids {
            let state = self.speakers.get_mut(&pseudo_id).expect("speaker state");
            if !state.residual.is_empty() {
                // zero-pad to a frame boundary so residual speech isn't lost.
                let mut padded = std::mem::take(&mut state.residual);
                let pad = FRAME_SIZE - padded.len();
                padded.extend(std::iter::repeat_n(0.0f32, pad));
                state.tracker.extend_pcm(&padded);
                let probs = state
                    .ctx
                    .process(&padded)
                    .map_err(|e| PipelineError::Vad(format!("engine finalize: {e}")))?;
                let frames_seen_before = state.tracker.frames_emitted;
                for (i, prob) in probs.into_iter().enumerate() {
                    let frame_idx = frames_seen_before + i as u32;
                    if let Some(region) = state.tracker.step(frame_idx, prob) {
                        all_regions.push(region);
                    }
                }
                state.tracker.frames_emitted += 1;
            }
            let total = state.tracker.frames_emitted;
            if let Some(region) = state.tracker.finalize(total) {
                all_regions.push(region);
            }
        }

        for region in all_regions {
            metrics::counter!(
                "chronicle_pipeline_operator_emit_count",
                "operator" => "vad",
            )
            .increment(1);
            self.pending.push(region);
        }

        Ok(std::mem::take(&mut self.pending))
    }

    fn name(&self) -> &'static str {
        "vad"
    }
}

// ---------------------------------------------------------------------------
// 48 kHz -> 16 kHz decimation
// ---------------------------------------------------------------------------

/// Simple 3:1 decimation with a 3-tap box filter. Good enough for VAD;
/// caller is expected to have already sent us clean mono PCM. We do NOT
/// use this for Whisper-grade resampling — that job belongs in the
/// transcription operator (or the caller).
pub(crate) fn downsample_48k_to_16k_i16(input: &[i16]) -> Vec<f32> {
    let out_len = input.len() / 3;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let base = i * 3;
        let a = input[base] as f32;
        let b = input[base + 1] as f32;
        let c = input[base + 2] as f32;
        let avg = (a + b + c) / 3.0 / i16::MAX as f32;
        out.push(avg);
    }
    out
}

// ---------------------------------------------------------------------------
// Silero ONNX engine (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "vad")]
pub mod silero;

// ---------------------------------------------------------------------------
// Test engine — deterministic probabilities for property tests.
// ---------------------------------------------------------------------------

/// Trivial engine that computes speech probability as the RMS of each
/// frame normalised to [0, 1]. Useful for testing the tracker state
/// machine without ONNX.
pub struct RmsEngine;

pub struct RmsContext;

impl VadEngine for RmsEngine {
    fn new_context(&self) -> Box<dyn VadContext> {
        Box::new(RmsContext)
    }
}

impl VadContext for RmsContext {
    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, PipelineError> {
        let frames = samples.len() / FRAME_SIZE;
        let mut probs = Vec::with_capacity(frames);
        for f in 0..frames {
            let frame = &samples[f * FRAME_SIZE..(f + 1) * FRAME_SIZE];
            let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
            probs.push(rms.clamp(0.0, 1.0));
        }
        Ok(probs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VadConfig;
    use uuid::Uuid;

    fn chunk_from_f32_48k(pcm_48k: &[f32], session_id: Uuid, pseudo_id: &str) -> AudioChunk {
        let pcm: Vec<i16> = pcm_48k
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();
        let duration_ms = ((pcm.len() as u64) * 1000 / 48_000) as u32;
        AudioChunk {
            session_id,
            pseudo_id: pseudo_id.into(),
            seq: 0,
            capture_started_at: 0,
            duration_ms,
            pcm: Arc::from(pcm),
        }
    }

    fn silence(ms: u32) -> Vec<f32> {
        vec![0.0; (48_000 * ms as usize) / 1000]
    }

    fn tone(ms: u32, freq: f32, amp: f32) -> Vec<f32> {
        let n = (48_000 * ms as usize) / 1000;
        (0..n)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                amp * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[tokio::test]
    async fn emits_region_at_tone_boundary() {
        let session = Uuid::new_v4();
        let mut pcm = silence(500);
        pcm.extend(tone(1000, 440.0, 0.5));
        pcm.extend(silence(1000));
        let chunk = chunk_from_f32_48k(&pcm, session, "spk");

        let mut op = VadOperator::new(
            VadConfig {
                threshold: 0.05,
                min_speech_ms: 100,
                min_silence_ms: 200,
                pad_ms: 50,
                model_path: None,
            },
            Box::new(RmsEngine),
        );

        op.ingest(chunk).await.unwrap();
        let finalized = op.finalize().await.unwrap();

        assert_eq!(finalized.len(), 1, "expected one region, got {finalized:?}");
        let r = &finalized[0];
        // Region should roughly match the tone window (450-1550 ms).
        assert!(r.start_ms >= 200 && r.start_ms <= 600, "start_ms = {}", r.start_ms);
        assert!(r.end_ms >= 1300 && r.end_ms <= 1700, "end_ms = {}", r.end_ms);
        assert!(!r.pcm.is_empty());
    }

    #[tokio::test]
    async fn silence_only_emits_nothing() {
        let session = Uuid::new_v4();
        let chunk = chunk_from_f32_48k(&silence(2000), session, "spk");

        let mut op = VadOperator::new(
            VadConfig::default(),
            Box::new(RmsEngine),
        );
        op.ingest(chunk).await.unwrap();
        let out = op.finalize().await.unwrap();
        assert!(out.is_empty(), "silence must not emit regions: {out:?}");
    }

    #[tokio::test]
    async fn streaming_equals_one_shot_regions() {
        // Same audio fed as 100 ms increments should yield the same
        // region boundaries as one shot.
        let session = Uuid::new_v4();
        let mut pcm = silence(300);
        pcm.extend(tone(800, 440.0, 0.5));
        pcm.extend(silence(600));
        pcm.extend(tone(400, 440.0, 0.5));
        pcm.extend(silence(300));

        let cfg = VadConfig {
            threshold: 0.05,
            min_speech_ms: 100,
            min_silence_ms: 200,
            pad_ms: 0,
            model_path: None,
        };

        // One shot
        let mut op1 = VadOperator::new(cfg.clone(), Box::new(RmsEngine));
        op1.ingest(chunk_from_f32_48k(&pcm, session, "spk")).await.unwrap();
        let one_shot = op1.finalize().await.unwrap();

        // Streaming at 100ms
        let mut op2 = VadOperator::new(cfg.clone(), Box::new(RmsEngine));
        let samples_per_chunk = 48_000 / 10; // 100ms
        for piece in pcm.chunks(samples_per_chunk) {
            op2.ingest(chunk_from_f32_48k(piece, session, "spk")).await.unwrap();
        }
        let streaming = op2.finalize().await.unwrap();

        assert_eq!(
            one_shot.len(),
            streaming.len(),
            "one_shot={one_shot:?} streaming={streaming:?}"
        );
        for (a, b) in one_shot.iter().zip(streaming.iter()) {
            assert!(
                (a.start_ms as i64 - b.start_ms as i64).abs() <= 40,
                "start drift a={} b={}",
                a.start_ms,
                b.start_ms
            );
            assert!(
                (a.end_ms as i64 - b.end_ms as i64).abs() <= 40,
                "end drift a={} b={}",
                a.end_ms,
                b.end_ms
            );
        }
    }
}
