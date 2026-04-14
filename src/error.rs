//! Pipeline error taxonomy.
//!
//! Per spec, errors split into two tiers:
//!
//! - **Per-input**: an operator catches, emits a [`DroppedRecord`] and
//!   continues. Those never surface as [`PipelineError`].
//! - **Pipeline-level**: bubble up as [`PipelineError`]. The caller tears
//!   down the `Pipeline`.
//!
//! [`DroppedRecord`]: crate::types::DroppedRecord

use thiserror::Error;

/// Single error enum for every pipeline-level failure. Callers match on
/// this or propagate with `?`.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("VAD error: {0}")]
    Vad(String),

    #[error("whisper error: {0}")]
    Whisper(#[from] WhisperError),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("config invalid: {0}")]
    ConfigInvalid(String),

    #[error("operator {operator} failed: {source}")]
    OperatorFailed {
        operator: &'static str,
        #[source]
        source: BoxError,
    },
}

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

impl PipelineError {
    pub fn operator_failed<E>(operator: &'static str, err: E) -> Self
    where
        E: Into<BoxError>,
    {
        Self::OperatorFailed {
            operator,
            source: err.into(),
        }
    }
}

/// Errors surfaced from the caller-supplied Whisper client. Recoverable
/// (retried within the transcription operator) unless `Fatal`.
#[derive(Debug, Error)]
pub enum WhisperError {
    #[error("whisper transient failure: {0}")]
    Transient(String),

    #[error("whisper fatal failure: {0}")]
    Fatal(String),
}

impl WhisperError {
    pub fn is_transient(&self) -> bool {
        matches!(self, WhisperError::Transient(_))
    }
}

pub type Result<T> = std::result::Result<T, PipelineError>;
