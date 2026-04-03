//! Core pipeline types.
//!
//! Data structures that flow between pipeline stages. Each stage takes
//! typed input and returns typed output — no shared mutable state.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single speaker's raw audio track as provided by the caller.
#[derive(Debug, Clone)]
pub struct SpeakerTrack {
    /// Opaque speaker identifier (e.g. "speaker_a").
    pub pseudo_id: String,
    /// Raw PCM bytes: signed 16-bit little-endian.
    pub pcm_data: Vec<u8>,
    /// Sample rate of the input audio, typically 48000 Hz from Discord.
    pub sample_rate: u32,
    /// Number of audio channels, typically 2 (stereo) from Discord.
    pub channels: u16,
}

/// Input to the pipeline: a session's worth of speaker tracks.
#[derive(Debug, Clone)]
pub struct SessionInput {
    /// Unique session identifier.
    pub session_id: Uuid,
    /// Per-speaker audio tracks to process.
    pub tracks: Vec<SpeakerTrack>,
}

/// Decoded mono f32 samples for a single speaker.
#[derive(Debug, Clone)]
pub struct SpeakerSamples {
    /// Speaker identifier, carried forward from `SpeakerTrack`.
    pub pseudo_id: String,
    /// Mono f32 audio samples in [-1.0, 1.0] range.
    pub samples: Vec<f32>,
    /// Sample rate of these samples.
    pub sample_rate: u32,
}

/// A contiguous region of detected speech within a speaker's audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRegion {
    /// Speaker who produced this speech.
    pub speaker: String,
    /// Start time in seconds (absolute within the session).
    pub start: f32,
    /// End time in seconds (absolute within the session).
    pub end: f32,
}

/// An extracted audio chunk ready for transcription.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Speaker who produced this audio.
    pub speaker: String,
    /// Mono f32 samples at 16kHz.
    pub samples: Vec<f32>,
    /// Sample rate (always 16000).
    pub sample_rate: u32,
    /// Absolute start time within the session, in seconds.
    pub original_start: f32,
    /// Absolute end time within the session, in seconds.
    pub original_end: f32,
}

/// A single transcript segment produced by Whisper and processed by filters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    /// Unique segment identifier.
    pub id: Uuid,
    /// Session this segment belongs to.
    pub session_id: Uuid,
    /// Ordering index within the session.
    pub segment_index: u32,
    /// Speaker who said this.
    pub speaker_pseudo_id: String,
    /// Absolute start time in seconds.
    pub start_time: f32,
    /// Absolute end time in seconds.
    pub end_time: f32,
    /// Text after filter processing. May differ from `original_text`.
    pub text: String,
    /// Immutable Whisper output, preserved for audit/debugging.
    pub original_text: String,
    /// Whisper confidence score, if available.
    pub confidence: Option<f32>,
    /// Scene/chunk group assigned by the scene chunker filter.
    pub chunk_group: Option<u32>,
    /// Whether this segment was excluded by a filter.
    pub excluded: bool,
    /// Reason for exclusion, if excluded.
    pub exclude_reason: Option<String>,
}

/// Final output of the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// All transcript segments (including excluded ones).
    pub segments: Vec<TranscriptSegment>,
    /// Number of segments that passed filters.
    pub segments_produced: u32,
    /// Number of segments excluded by filters.
    pub segments_excluded: u32,
    /// Number of scenes detected by the scene chunker.
    pub scenes_detected: u32,
    /// Total audio duration processed, in seconds.
    pub duration_processed: f32,
}
