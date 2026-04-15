//! The `Operator` trait: ingest / emit / finalize.
//!
//! Every stage implements this. The `Pipeline` composes them in order
//! and adapts the output of one into the input of the next.

use async_trait::async_trait;

use crate::error::PipelineError;

/// A single stage of the pipeline. See module docs for the lifecycle.
///
/// Operators are single-threaded, own their own state, and never mutate
/// shared state across boundaries. The framework polls them sequentially
/// per input.
#[async_trait]
pub trait Operator: Send + 'static {
    type Input: Send + 'static;
    type Output: Send + 'static;

    /// Consume a single input. May update internal state; may not need
    /// to produce output immediately (e.g. VAD accumulates frames).
    async fn ingest(&mut self, input: Self::Input) -> Result<(), PipelineError>;

    /// Drain whatever output is ready. Called between ingests. Must not
    /// block waiting for more input. Pure state-transfer — no I/O.
    fn emit(&mut self) -> Vec<Self::Output>;

    /// End-of-input. Flush open state (e.g. an ongoing VAD region) and
    /// return any final outputs.
    async fn finalize(&mut self) -> Result<Vec<Self::Output>, PipelineError>;

    /// Static name for tracing / metrics. Must be stable across runs.
    fn name(&self) -> &'static str;
}
