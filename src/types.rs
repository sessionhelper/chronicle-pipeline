//! Pipeline public types.
//!
//! Every type the caller constructs or receives lives here. Operators
//! and the pipeline compose these; they never invent their own
//! cross-stage types.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Session identifier. Opaque to the pipeline; the caller asserts meaning.
pub type SessionId = Uuid;

/// Speaker pseudonym. Opaque to the pipeline.
pub type PseudoId = String;

/// Wall-clock timestamp associated with an audio chunk's capture start.
/// Microsecond precision is sufficient for synchronising across speakers.
pub type Timestamp = i64;

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// A single chunk of per-speaker PCM audio arriving at the pipeline.
///
/// The pipeline expects **48 kHz mono s16le** — the caller is responsible
/// for stereo downmix and sample-rate conversion before constructing
/// this. `seq` is the caller's monotonic per-speaker sequence number;
/// the pipeline uses it only for tracing.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub session_id: SessionId,
    pub pseudo_id: PseudoId,
    pub seq: u32,
    pub capture_started_at: Timestamp,
    pub duration_ms: u32,
    pub pcm: Arc<[i16]>,
}

impl AudioChunk {
    pub const SAMPLE_RATE: u32 = 48_000;

    pub fn sample_count(&self) -> usize {
        self.pcm.len()
    }
}

/// A full session's audio, used in one-shot mode. Equivalent to a
/// sequence of `AudioChunk`s per pseudo_id.
#[derive(Debug, Clone)]
pub struct SessionAudio {
    pub session_id: SessionId,
    pub tracks: Vec<SessionTrack>,
}

#[derive(Debug, Clone)]
pub struct SessionTrack {
    pub pseudo_id: PseudoId,
    pub capture_started_at: Timestamp,
    pub pcm: Arc<[i16]>,
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

/// A canonical transcript segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Segment {
    pub id: Uuid,
    pub session_id: SessionId,
    pub pseudo_id: PseudoId,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    /// Immutable raw Whisper text captured at segment creation.
    pub original: String,
    pub confidence: f32,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub flags: SegmentFlags,
}

/// Per-segment flags. Additive — new flags tolerated by old consumers
/// via `#[serde(default)]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SegmentFlags {
    #[serde(default)]
    pub meta_talk: Option<MetaTalkLabel>,
}

/// Meta-talk classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetaTalkLabel {
    InCharacter,
    OutOfCharacter,
    Mixed,
    Unclear,
}

/// A narrative beat.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Beat {
    pub id: Uuid,
    pub session_id: SessionId,
    pub t_ms: u64,
    pub kind: BeatKind,
    pub label: String,
    pub confidence: f32,
}

/// Closed, versioned beat kinds. `Unknown(String)` is the serde fallthrough
/// so old consumers don't explode when new variants appear.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeatKind {
    CombatStart,
    CombatEnd,
    Discovery,
    DialogueClimax,
    SceneBreak,
    #[serde(other)]
    Unknown,
}

/// A coarse-grained scene groping multiple beats.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Scene {
    pub id: Uuid,
    pub session_id: SessionId,
    pub start_ms: u64,
    pub end_ms: u64,
    pub label: String,
    pub confidence: f32,
}

/// A record representing a per-input failure in some operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DroppedRecord {
    pub source_operator: String,
    pub reason: DropReason,
    #[serde(default)]
    pub details: serde_json::Value,
}

/// Closed enum for drop reasons. Add variants when new drop sources appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DropReason {
    /// VAD-produced region was malformed or empty.
    InvalidVadRegion,
    /// Whisper exhausted retry budget for one region.
    WhisperExhaustedRetries,
    /// Whisper returned an empty / invalid transcription payload.
    WhisperBadPayload,
    /// Transcription matched a known hallucination heuristic.
    Hallucination,
    /// Too short / too noisy to keep.
    NoiseFilter,
    /// Catch-all for heuristic per-input rejections.
    HeuristicReject,
}

/// Pipeline-level output. Returned from `emit()` (drainable) and from
/// `finalize()` / `run_one_shot()` (complete).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PipelineOutput {
    pub segments: Vec<Segment>,
    pub beats: Vec<Beat>,
    pub scenes: Vec<Scene>,
    pub dropped: Vec<DroppedRecord>,
}

impl PipelineOutput {
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
            && self.beats.is_empty()
            && self.scenes.is_empty()
            && self.dropped.is_empty()
    }

    pub fn extend(&mut self, other: PipelineOutput) {
        self.segments.extend(other.segments);
        self.beats.extend(other.beats);
        self.scenes.extend(other.scenes);
        self.dropped.extend(other.dropped);
    }

    /// Sort segments / beats / scenes by start time. Called once at
    /// pipeline emit boundaries.
    pub fn sort_in_place(&mut self) {
        self.segments.sort_by_key(|s| s.start_ms);
        self.beats.sort_by_key(|b| b.t_ms);
        self.scenes.sort_by_key(|s| s.start_ms);
    }
}

// ---------------------------------------------------------------------------
// Inter-operator message types
// ---------------------------------------------------------------------------

/// A closed voice region emitted by the VAD operator.
#[derive(Debug, Clone)]
pub struct VoiceRegion {
    pub session_id: SessionId,
    pub pseudo_id: PseudoId,
    pub start_ms: u64,
    pub end_ms: u64,
    /// Mono 16 kHz f32 audio covering the region, suitable for Whisper.
    pub pcm: Arc<[f32]>,
}

/// A transcription result tied to a region.
#[derive(Debug, Clone)]
pub struct TranscribedRegion {
    pub session_id: SessionId,
    pub pseudo_id: PseudoId,
    pub start_ms: u64,
    pub end_ms: u64,
    pub transcription: Transcription,
}

/// Whisper output. Callable-supplied via `WhisperClient`.
#[derive(Debug, Clone)]
pub struct Transcription {
    pub text: String,
    pub confidence: f32,
    pub language: Option<String>,
}

/// A segment that has been filtered and is ready for downstream operators
/// (meta-talk, beats, scenes). Wrapped so operators can distinguish
/// "just-emitted" from later passes.
#[derive(Debug, Clone)]
pub enum DownstreamItem {
    Segment(Segment),
    Beat(Beat),
}
