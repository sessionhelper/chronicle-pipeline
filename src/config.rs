//! Pipeline configuration.
//!
//! All knobs for every operator live in one place so the caller can
//! deserialize once from TOML/JSON and pass it in.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Top-level
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PipelineConfig {
    /// Ordered operator chain. The pipeline builder validates this is
    /// a legal ordering (VAD before Transcription, Segment before
    /// MetaTalk, etc.) and errors out otherwise.
    #[serde(default = "default_operators")]
    pub operators: Vec<OperatorKind>,

    #[serde(default)]
    pub vad: VadConfig,
    #[serde(default)]
    pub transcription: TranscriptionConfig,
    #[serde(default)]
    pub filter: FilterConfig,
    #[serde(default)]
    pub meta_talk: MetaTalkConfig,
    #[serde(default)]
    pub beats: BeatsConfig,
    #[serde(default)]
    pub scenes: ScenesConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            operators: default_operators(),
            vad: VadConfig::default(),
            transcription: TranscriptionConfig::default(),
            filter: FilterConfig::default(),
            meta_talk: MetaTalkConfig::default(),
            beats: BeatsConfig::default(),
            scenes: ScenesConfig::default(),
        }
    }
}

fn default_operators() -> Vec<OperatorKind> {
    vec![
        OperatorKind::Vad,
        OperatorKind::Transcription,
        OperatorKind::Filter,
        OperatorKind::Segment,
        OperatorKind::MetaTalk,
        OperatorKind::Beats,
        OperatorKind::Scenes,
    ]
}

/// Closed enum of operator kinds. Declarative ordering via a `Vec`
/// of these, validated by the builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorKind {
    Vad,
    Transcription,
    Filter,
    Segment,
    MetaTalk,
    Beats,
    Scenes,
}

impl OperatorKind {
    pub const ALL: &'static [OperatorKind] = &[
        OperatorKind::Vad,
        OperatorKind::Transcription,
        OperatorKind::Filter,
        OperatorKind::Segment,
        OperatorKind::MetaTalk,
        OperatorKind::Beats,
        OperatorKind::Scenes,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            OperatorKind::Vad => "vad",
            OperatorKind::Transcription => "transcription",
            OperatorKind::Filter => "filter",
            OperatorKind::Segment => "segment",
            OperatorKind::MetaTalk => "meta_talk",
            OperatorKind::Beats => "beats",
            OperatorKind::Scenes => "scenes",
        }
    }
}

// ---------------------------------------------------------------------------
// Per-operator config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct VadConfig {
    pub threshold: f32,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub pad_ms: u32,
    /// Path to the Silero ONNX model. Required when the `vad` feature
    /// is enabled and the caller uses the default engine. `None` means
    /// the caller must inject a custom engine.
    #[serde(default)]
    pub model_path: Option<PathBuf>,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 800,
            pad_ms: 100,
            model_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TranscriptionConfig {
    /// Max retry attempts (spec says 3).
    pub max_attempts: u32,
    /// Initial backoff (ms). Doubles each attempt.
    pub initial_backoff_ms: u64,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 500,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FilterConfig {
    /// Minimum confidence to keep a transcription.
    pub min_confidence: f32,
    /// Maximum characters per second — higher means impossibly-fast
    /// speech, usually a Whisper artefact.
    pub max_cps: f32,
    /// Minimum alphabetic character count. Ditches pure punctuation
    /// or single-character noise.
    pub min_alpha_chars: usize,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            min_confidence: -1.5,
            max_cps: 25.0,
            min_alpha_chars: 2,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MetaTalkConfig {}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct BeatsConfig {
    /// Minimum silence (ms) to mark a scene-break-adjacent beat.
    pub scene_break_silence_ms: u64,
    /// Minimum segment confidence for the beat detector to consider it.
    pub min_segment_confidence: f32,
}

impl Default for BeatsConfig {
    fn default() -> Self {
        Self {
            scene_break_silence_ms: 8_000,
            min_segment_confidence: -1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ScenesConfig {
    /// Silence gap (ms) between segments that starts a new scene.
    pub max_silence_gap_ms: u64,
    /// Maximum scene duration (ms).
    pub max_scene_ms: u64,
}

impl Default for ScenesConfig {
    fn default() -> Self {
        Self {
            max_silence_gap_ms: 30_000,
            max_scene_ms: 600_000,
        }
    }
}
