//! Filter chain for post-transcription processing.
//!
//! Filters implement the `StreamFilter` trait and are applied in order
//! to each transcript segment. The crate ships with hallucination
//! detection and scene chunking.

pub mod hallucination;
pub mod scene_chunker;

use crate::error::{PipelineError, Result};
use crate::types::TranscriptSegment;

/// Result of applying a filter to a single segment.
#[derive(Debug, Clone)]
pub enum FilterResult {
    /// Segment passes this filter.
    Pass,
    /// Segment should be excluded, with a reason.
    Exclude {
        /// Human-readable explanation of why the segment was excluded.
        reason: String,
    },
}

/// Trait for pluggable transcript filters.
///
/// Each filter maintains internal state and processes segments one at a time
/// via `on_segment`. Periodic `sweep` calls allow retroactive analysis
/// (e.g. frequency-based hallucination detection). `finalize` is called
/// once after all segments are processed.
#[async_trait::async_trait]
pub trait StreamFilter: Send + Sync {
    /// Process a single segment. May mutate the segment (e.g. set `chunk_group`)
    /// or return `Exclude` to mark it for removal.
    async fn on_segment(&mut self, segment: &mut TranscriptSegment) -> FilterResult;

    /// Periodic sweep for retroactive analysis. Returns the number of
    /// segments retroactively excluded.
    async fn sweep(&mut self) -> Result<u32>;

    /// Called once after all segments have been processed. Perform any
    /// final cleanup or retroactive marking.
    async fn finalize(&mut self) -> Result<()>;
}

/// Apply all filters to a list of transcript segments.
///
/// Each segment passes through every filter in order. If any filter
/// returns `Exclude`, the segment is marked as excluded with the reason.
/// After all segments, `sweep` and `finalize` are called on each filter.
pub async fn apply_filters(
    mut segments: Vec<TranscriptSegment>,
    filters: &mut [Box<dyn StreamFilter>],
) -> Result<Vec<TranscriptSegment>> {
    // Run each segment through all filters
    for segment in segments.iter_mut() {
        for filter in filters.iter_mut() {
            if segment.excluded {
                break; // Already excluded by a previous filter
            }

            match filter.on_segment(segment).await {
                FilterResult::Pass => {}
                FilterResult::Exclude { reason } => {
                    segment.excluded = true;
                    segment.exclude_reason = Some(reason);
                }
            }
        }
    }

    // Run sweep on each filter for retroactive analysis
    for filter in filters.iter_mut() {
        filter
            .sweep()
            .await
            .map_err(|e| PipelineError::Filter(e.to_string()))?;
    }

    // Finalize each filter
    for filter in filters.iter_mut() {
        filter
            .finalize()
            .await
            .map_err(|e| PipelineError::Filter(e.to_string()))?;
    }

    tracing::info!(
        total = segments.len(),
        excluded = segments.iter().filter(|s| s.excluded).count(),
        "filter chain complete"
    );

    Ok(segments)
}

/// Create the default filter chain: hallucination detection + scene chunking.
pub fn default_filters() -> Vec<Box<dyn StreamFilter>> {
    vec![
        Box::new(hallucination::HallucinationFilter::new()),
        Box::new(scene_chunker::SceneChunker::new(
            scene_chunker::SceneChunkerConfig::default(),
        )),
    ]
}
