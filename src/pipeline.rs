//! Pipeline orchestration.
//!
//! Chains streaming stages: resample → RMS audio detection →
//! Silero VAD → Whisper transcription → filters. Input is mono f32
//! samples per speaker — byte decoding and downmix are the caller's
//! responsibility.

use crate::ad::{self, RmsConfig};
use crate::audio::resample;
use crate::error::Result;
use crate::filters::{self, StreamFilter};
use crate::transcribe::{self, TranscriberConfig};
use crate::types::{PipelineResult, SessionInput};
use crate::vad::{self, VadConfig};

/// Top-level pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// RMS audio detection (tier 1 — silence gate).
    pub rms: RmsConfig,
    /// VAD configuration (tier 2 — speech vs non-speech).
    pub vad: VadConfig,
    /// Whisper transcription configuration.
    pub whisper: TranscriberConfig,
    /// Minimum speech region duration in seconds. Regions shorter
    /// than this are dropped before transcription — the "pop filter"
    /// that catches blips, breaths, and mic noise pre-Whisper.
    pub min_chunk_duration: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            rms: RmsConfig::default(),
            vad: VadConfig::default(),
            whisper: TranscriberConfig {
                endpoint: "http://localhost:8080/v1/audio/transcriptions".into(),
                model: "large-v3-turbo".into(),
                language: Some("en".into()),
            },
            min_chunk_duration: 0.8,
        }
    }
}

/// Process a completed recording session end-to-end.
///
/// Audio flows through: resample (to 16kHz) → RMS audio detection
/// (silence removed) → Silero VAD (non-speech removed) → Whisper
/// (speech → text) → filters (hallucination detection, scene chunking).
/// Input is already mono f32 samples — caller handles byte decoding.
pub async fn process_session(
    config: &PipelineConfig,
    input: SessionInput,
    filters: &mut [Box<dyn StreamFilter>],
) -> Result<PipelineResult> {
    let session_id = input.session_id;

    // Resample to 16kHz for VAD and Whisper.
    // Input is already mono f32 — caller handled byte decoding and downmix.
    let input_rate = input.tracks.first().map(|t| t.sample_rate).unwrap_or(48000);
    let resampled = resample::resample(&input.tracks, input_rate, 16000)?;

    let duration_processed: f32 = resampled
        .iter()
        .map(|s| s.samples.len() as f32 / s.sample_rate as f32)
        .sum();

    // 3. RMS audio detection — silence gate (tier 1).
    //    Removes dead air before it reaches VAD.
    let audio_segments = ad::detect_audio_all(&config.rms, &resampled);

    // 4. Silero VAD — speech vs non-speech (tier 2).
    //    Catches breaths, keyboard, mic bumps that passed RMS.
    let voice_chunks =
        vad::detect_speech_from_segments(&config.vad, &audio_segments).await?;

    tracing::info!(
        speakers = input.tracks.len(),
        duration_processed,
        rms_segments = audio_segments.len(),
        voice_chunks = voice_chunks.len(),
        "audio detection complete, sending speech to Whisper"
    );

    // 5. Transcribe speech chunks via Whisper.
    let segments =
        transcribe::transcribe(&config.whisper, &voice_chunks, session_id).await?;

    // 6. Filter chain (hallucination detection, scene chunking).
    let filtered = filters::apply_filters(segments, filters).await?;

    let segments_produced = filtered.iter().filter(|s| !s.excluded).count() as u32;
    let segments_excluded = filtered.iter().filter(|s| s.excluded).count() as u32;
    let scenes_detected = filtered
        .iter()
        .filter(|s| !s.excluded)
        .filter_map(|s| s.chunk_group)
        .max()
        .map(|max| max + 1)
        .unwrap_or(0);

    tracing::info!(
        session_id = %session_id,
        segments_produced,
        segments_excluded,
        scenes_detected,
        duration_processed,
        "pipeline complete"
    );

    Ok(PipelineResult {
        segments: filtered,
        segments_produced,
        segments_excluded,
        scenes_detected,
        duration_processed,
    })
}
