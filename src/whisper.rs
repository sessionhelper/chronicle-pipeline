//! Caller-supplied Whisper client abstraction.
//!
//! The pipeline never constructs an HTTP client. It holds an
//! `Arc<dyn WhisperClient>` and calls `.transcribe()` through it.

use async_trait::async_trait;
use std::sync::Arc;

use crate::error::WhisperError;
use crate::types::Transcription;

#[async_trait]
pub trait WhisperClient: Send + Sync + 'static {
    /// Transcribe mono 16 kHz f32 PCM. Implementations decide their
    /// own retry / timeout policy; the pipeline layers its own retries
    /// on top (see [`crate::operators::transcription`]).
    async fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Transcription, WhisperError>;
}

/// Bundle of caller-supplied resources. Plugged into the pipeline via
/// [`crate::pipeline::PipelineBuilder::deps`].
#[derive(Clone)]
pub struct PipelineDeps {
    pub whisper: Arc<dyn WhisperClient>,
    #[cfg(feature = "vad")]
    pub vad_engine: Option<Arc<dyn VadEngineFactory>>,
}

impl std::fmt::Debug for PipelineDeps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineDeps").finish_non_exhaustive()
    }
}

/// A factory that mints fresh VAD engine instances. Optional; if `None`,
/// the builder instantiates the default Silero engine from
/// [`crate::config::VadConfig::model_path`] when the `vad` feature
/// is enabled.
#[cfg(feature = "vad")]
pub trait VadEngineFactory: Send + Sync + 'static {
    fn build(&self) -> Box<dyn crate::operators::vad::VadEngine>;
}
