//! chronicle-pipeline: structured session understanding from per-speaker PCM.
//!
//! See `/home/alex/sessionhelper/sessionhelper-hub/docs/modules/chronicle-pipeline.md`
//! for the authoritative module spec.
//!
//! # Usage
//!
//! ```no_run
//! use chronicle_pipeline::{
//!     config::PipelineConfig, pipeline::Pipeline, whisper::WhisperClient,
//!     types::Transcription, error::WhisperError,
//! };
//! use async_trait::async_trait;
//! use std::sync::Arc;
//!
//! struct MyWhisper;
//!
//! #[async_trait]
//! impl WhisperClient for MyWhisper {
//!     async fn transcribe(&self, _audio: &[f32], _sr: u32)
//!         -> Result<Transcription, WhisperError>
//!     { unimplemented!() }
//! }
//!
//! # async fn _demo() {
//! let cfg = PipelineConfig::default();
//! let pipeline = Pipeline::builder(cfg)
//!     .whisper(Arc::new(MyWhisper))
//!     .build()
//!     .unwrap();
//! // drive it with `ingest_chunk` / `emit` / `finalize`, or `run_one_shot`.
//! # }
//! ```

pub mod config;
pub mod error;
pub mod operator;
pub mod operators;
pub mod pipeline;
pub mod types;
pub mod whisper;

pub use config::{
    BeatsConfig, FilterConfig, MetaTalkConfig, OperatorKind, PipelineConfig, ScenesConfig,
    TranscriptionConfig, VadConfig,
};
pub use error::{PipelineError, Result, WhisperError};
pub use operator::Operator;
pub use pipeline::{Pipeline, PipelineBuilder};
pub use types::{
    AudioChunk, Beat, BeatKind, DropReason, DroppedRecord, MetaTalkLabel, PipelineOutput,
    PseudoId, Scene, Segment, SegmentFlags, SessionAudio, SessionId, SessionTrack, Timestamp,
    Transcription,
};
pub use whisper::{PipelineDeps, WhisperClient};
