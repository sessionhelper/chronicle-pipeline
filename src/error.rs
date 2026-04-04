//! Pipeline error types.
//!
//! Single error enum for the whole crate. Each stage maps its internal
//! errors via `?` so the caller gets one type to match on.

/// Errors that can occur during pipeline processing.
#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    /// Failed to resample audio (e.g. 48kHz -> 16kHz).
    #[error("resample error: {0}")]
    Resample(String),

    /// Voice activity detection failed.
    #[error("VAD error: {0}")]
    Vad(String),

    /// Whisper transcription failed.
    #[error("transcription error: {0}")]
    Transcribe(String),

    /// Filter stage error.
    #[error("filter error: {0}")]
    Filter(String),

    /// HTTP request failed (feature-gated behind `transcribe`).
    #[cfg(feature = "transcribe")]
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, PipelineError>;
