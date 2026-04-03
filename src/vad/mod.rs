//! Voice activity detection via Silero VAD (ONNX runtime).
//!
//! Processes 16kHz mono audio and returns speech regions per speaker.
//! When the `vad` feature is disabled, falls back to treating all
//! audio as a single speech region (no filtering).

#[cfg(feature = "vad")]
use crate::error::PipelineError;
use crate::error::Result;
use crate::types::{SpeakerSamples, SpeechRegion};

/// Configuration for the Silero VAD model.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Path to the Silero VAD ONNX model file.
    pub model_path: std::path::PathBuf,
    /// Speech probability threshold (0.0 - 1.0). Frames above this
    /// are considered speech.
    pub threshold: f32,
    /// Minimum duration of a speech region in seconds. Regions shorter
    /// than this are discarded as noise.
    pub min_speech_duration: f32,
    /// Minimum silence duration in seconds required to split two
    /// speech regions apart.
    pub min_silence_duration: f32,
    /// Padding in seconds added to both sides of each speech region
    /// to avoid clipping word boundaries.
    pub speech_pad: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("silero_vad.onnx"),
            threshold: 0.5,
            min_speech_duration: 0.25,
            min_silence_duration: 0.8,
            speech_pad: 0.1,
        }
    }
}

/// Run voice activity detection on all speaker samples.
///
/// Returns speech regions sorted chronologically across all speakers.
/// Each region identifies who is speaking and the time boundaries.
pub async fn detect_speech(
    config: &VadConfig,
    samples: &[SpeakerSamples],
) -> Result<Vec<SpeechRegion>> {
    let mut all_regions = Vec::new();

    for speaker in samples {
        let regions = detect_speech_for_speaker(config, speaker)?;
        all_regions.extend(regions);
    }

    // Sort all regions chronologically by start time
    all_regions.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));

    tracing::info!(
        total_regions = all_regions.len(),
        speakers = samples.len(),
        "VAD complete"
    );

    Ok(all_regions)
}

/// Run VAD on a single speaker's samples.
#[cfg(feature = "vad")]
fn detect_speech_for_speaker(
    config: &VadConfig,
    speaker: &SpeakerSamples,
) -> Result<Vec<SpeechRegion>> {
    if speaker.sample_rate != 16000 {
        return Err(PipelineError::Vad(format!(
            "speaker {}: VAD expects 16kHz audio but got {}Hz",
            speaker.pseudo_id, speaker.sample_rate
        )));
    }

    // TODO: Load Silero VAD ONNX model via `ort` and run inference.
    //
    // Implementation outline:
    // 1. Load model from config.model_path using ort::Session
    // 2. Process audio in 30ms frames (480 samples at 16kHz)
    // 3. For each frame, run inference to get speech probability
    // 4. Threshold probabilities to detect speech/silence transitions
    // 5. Merge adjacent speech frames into regions
    // 6. Apply min_speech_duration, min_silence_duration, speech_pad
    // 7. Drop regions shorter than min_speech_duration
    //
    // For now, fall back to the stub behavior.

    let _ = config;
    let duration = speaker.samples.len() as f32 / speaker.sample_rate as f32;

    tracing::warn!(
        speaker = %speaker.pseudo_id,
        "VAD model not yet implemented, returning full audio as speech"
    );

    Ok(vec![SpeechRegion {
        speaker: speaker.pseudo_id.clone(),
        start: 0.0,
        end: duration,
    }])
}

/// Stub VAD when the `vad` feature is disabled.
/// Returns all audio as one big speech region per speaker.
#[cfg(not(feature = "vad"))]
fn detect_speech_for_speaker(
    _config: &VadConfig,
    speaker: &SpeakerSamples,
) -> Result<Vec<SpeechRegion>> {
    let duration = speaker.samples.len() as f32 / speaker.sample_rate as f32;

    tracing::info!(
        speaker = %speaker.pseudo_id,
        duration,
        "VAD disabled, treating full track as speech"
    );

    Ok(vec![SpeechRegion {
        speaker: speaker.pseudo_id.clone(),
        start: 0.0,
        end: duration,
    }])
}

/// Extract audio chunks from speaker samples based on speech regions.
///
/// Each speech region maps to an `AudioChunk` containing the corresponding
/// slice of 16kHz mono audio. Regions shorter than `min_chunk_duration`
/// are dropped.
pub fn extract_chunks(
    samples: &[SpeakerSamples],
    regions: &[SpeechRegion],
    min_chunk_duration: f32,
) -> Result<Vec<crate::types::AudioChunk>> {
    let chunks: Vec<crate::types::AudioChunk> = regions
        .iter()
        .filter(|r| (r.end - r.start) >= min_chunk_duration)
        .filter_map(|region| {
            // Find the matching speaker's samples
            let speaker = samples
                .iter()
                .find(|s| s.pseudo_id == region.speaker)?;

            let start_sample = (region.start * speaker.sample_rate as f32) as usize;
            let end_sample = (region.end * speaker.sample_rate as f32) as usize;

            // Clamp to valid range
            let start_sample = start_sample.min(speaker.samples.len());
            let end_sample = end_sample.min(speaker.samples.len());

            if start_sample >= end_sample {
                return None;
            }

            Some(crate::types::AudioChunk {
                speaker: region.speaker.clone(),
                samples: speaker.samples[start_sample..end_sample].to_vec(),
                sample_rate: speaker.sample_rate,
                original_start: region.start,
                original_end: region.end,
            })
        })
        .collect();

    tracing::debug!(
        regions = regions.len(),
        chunks = chunks.len(),
        min_chunk_duration,
        "extracted audio chunks"
    );

    Ok(chunks)
}
