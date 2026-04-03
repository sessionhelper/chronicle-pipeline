//! Pipeline orchestration.
//!
//! The main `process_session` function chains all stages with `?`.
//! Each step is a plain function call — the happy path reads
//! top-to-bottom like pseudocode.

use crate::audio::{decode, resample};
use crate::error::Result;
use crate::filters::{self, StreamFilter};
use crate::transcribe::{self, TranscriberConfig};
use crate::types::{PipelineResult, SessionInput};
use crate::vad::{self, VadConfig};

/// Top-level pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// VAD configuration.
    pub vad: VadConfig,
    /// Whisper transcription configuration.
    pub whisper: TranscriberConfig,
    /// Minimum speech region duration in seconds. Regions shorter
    /// than this are dropped before transcription.
    pub min_chunk_duration: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
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
/// Takes raw PCM audio per speaker, runs VAD, transcribes via Whisper,
/// applies filters, and returns structured transcript segments.
///
/// Each stage is a plain function chained with `?` — the compiler
/// enforces that we don't skip steps or reorder them.
pub async fn process_session(
    config: &PipelineConfig,
    input: SessionInput,
    filters: &mut [Box<dyn StreamFilter>],
) -> Result<PipelineResult> {
    let session_id = input.session_id;

    // Decode raw s16le PCM bytes into mono f32 samples
    let samples = decode::decode(&input.tracks)?;

    // Resample from input rate (48kHz) to 16kHz for Whisper/VAD
    let resampled = resample::resample(&samples, 48000, 16000)?;

    // Calculate total audio duration for reporting
    let duration_processed = resampled
        .iter()
        .map(|s| s.samples.len() as f32 / s.sample_rate as f32)
        .sum::<f32>();

    // Run voice activity detection to find speech regions
    let regions = vad::detect_speech(&config.vad, &resampled).await?;

    // Extract audio chunks from speech regions
    let chunks = vad::extract_chunks(&resampled, &regions, config.min_chunk_duration)?;

    // Transcribe each chunk via the external Whisper endpoint
    let segments = transcribe::transcribe(&config.whisper, &chunks, session_id).await?;

    // Apply filter chain (hallucination detection, scene chunking)
    let filtered = filters::apply_filters(segments, filters).await?;

    // Compute result stats
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
