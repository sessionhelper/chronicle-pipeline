//! Segment operator.
//!
//! Consumes filtered [`TranscribedRegion`]s + passthrough drops, produces
//! canonical [`Segment`]s with v7 UUIDs. The `original` field is captured
//! here at first write and never mutated.

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::PipelineError;
use crate::operator::Operator;
use crate::operators::filter::FilterOut;
use crate::types::{DroppedRecord, Segment, SegmentFlags};

/// Output of the segment operator. Same ordering as the input;
/// downstream operators dispatch on the variant.
#[derive(Debug, Clone)]
pub enum SegmentOut {
    Segment(Segment),
    Dropped(DroppedRecord),
}

pub struct SegmentOperator {
    pending: Vec<SegmentOut>,
}

impl SegmentOperator {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }
}

impl Default for SegmentOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Operator for SegmentOperator {
    type Input = FilterOut;
    type Output = SegmentOut;

    async fn ingest(&mut self, input: FilterOut) -> Result<(), PipelineError> {
        let span = tracing::info_span!("operator", name = "segment");
        let _enter = span.enter();

        match input {
            FilterOut::Dropped(d) => {
                self.pending.push(SegmentOut::Dropped(d));
            }
            FilterOut::Kept(region) => {
                let text = region.transcription.text.trim().to_string();
                let segment = Segment {
                    id: Uuid::now_v7(),
                    session_id: region.session_id,
                    pseudo_id: region.pseudo_id,
                    start_ms: region.start_ms,
                    end_ms: region.end_ms,
                    text: text.clone(),
                    original: text,
                    confidence: region.transcription.confidence,
                    language: region.transcription.language,
                    flags: SegmentFlags::default(),
                };
                metrics::counter!(
                    "chronicle_pipeline_operator_emit_count",
                    "operator" => "segment",
                )
                .increment(1);
                self.pending.push(SegmentOut::Segment(segment));
            }
        }
        Ok(())
    }

    fn emit(&mut self) -> Vec<SegmentOut> {
        std::mem::take(&mut self.pending)
    }

    async fn finalize(&mut self) -> Result<Vec<SegmentOut>, PipelineError> {
        Ok(std::mem::take(&mut self.pending))
    }

    fn name(&self) -> &'static str {
        "segment"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TranscribedRegion, Transcription};
    use uuid::Uuid;

    fn kept(text: &str, start: u64, end: u64) -> FilterOut {
        FilterOut::Kept(TranscribedRegion {
            session_id: Uuid::new_v4(),
            pseudo_id: "spk".into(),
            start_ms: start,
            end_ms: end,
            transcription: Transcription {
                text: text.into(),
                confidence: -0.1,
                language: Some("en".into()),
            },
        })
    }

    #[tokio::test]
    async fn builds_segment_with_v7_uuid_and_original() {
        let mut op = SegmentOperator::new();
        op.ingest(kept("hello world", 100, 500)).await.unwrap();
        let out = op.emit();
        match &out[0] {
            SegmentOut::Segment(s) => {
                assert_eq!(s.text, "hello world");
                assert_eq!(s.original, "hello world");
                assert_eq!(s.start_ms, 100);
                assert_eq!(s.end_ms, 500);
                // v7 UUIDs have version 7.
                assert_eq!(s.id.get_version_num(), 7);
            }
            _ => panic!("expected segment"),
        }
    }
}
