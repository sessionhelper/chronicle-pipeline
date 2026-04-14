//! Filter operator.
//!
//! Consumes [`TranscriptionOut`], drops hallucinations / noise
//! transcriptions, forwards keepers as [`TranscribedRegion`]. Drops
//! pass through as [`DroppedRecord`]s attached to [`FilterOut`].

use async_trait::async_trait;

use crate::config::FilterConfig;
use crate::error::PipelineError;
use crate::operator::Operator;
use crate::operators::transcription::TranscriptionOut;
use crate::types::{DropReason, DroppedRecord, TranscribedRegion};

/// Output of the filter operator.
#[derive(Debug, Clone)]
pub enum FilterOut {
    Kept(TranscribedRegion),
    Dropped(DroppedRecord),
}

pub struct FilterOperator {
    config: FilterConfig,
    pending: Vec<FilterOut>,
}

impl FilterOperator {
    pub fn new(config: FilterConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
        }
    }

    fn classify(&self, region: &TranscribedRegion) -> Classification {
        let text = region.transcription.text.trim();
        let conf = region.transcription.confidence;
        let duration_ms = region.end_ms.saturating_sub(region.start_ms) as f32;
        let duration_s = (duration_ms / 1000.0).max(0.1);
        let cps = text.chars().count() as f32 / duration_s;
        let alpha = text.chars().filter(|c| c.is_alphabetic()).count();

        if text.is_empty() {
            return Classification::Drop(DropReason::NoiseFilter, "empty text");
        }
        if alpha < self.config.min_alpha_chars {
            return Classification::Drop(DropReason::NoiseFilter, "no_alpha");
        }
        if is_repeated_phrase(text) {
            return Classification::Drop(DropReason::Hallucination, "repeated_phrase");
        }
        if is_known_hallucination(text) {
            return Classification::Drop(DropReason::Hallucination, "known_phrase");
        }
        if conf < self.config.min_confidence {
            return Classification::Drop(DropReason::NoiseFilter, "low_confidence");
        }
        if cps > self.config.max_cps {
            return Classification::Drop(DropReason::Hallucination, "impossible_cps");
        }
        Classification::Keep
    }
}

#[derive(Debug)]
enum Classification {
    Keep,
    Drop(DropReason, &'static str),
}

#[async_trait]
impl Operator for FilterOperator {
    type Input = TranscriptionOut;
    type Output = FilterOut;

    async fn ingest(&mut self, input: TranscriptionOut) -> Result<(), PipelineError> {
        let span = tracing::info_span!("operator", name = "filter");
        let _enter = span.enter();

        match input {
            TranscriptionOut::Dropped(d) => {
                self.pending.push(FilterOut::Dropped(d));
            }
            TranscriptionOut::Ok(region) => match self.classify(&region) {
                Classification::Keep => {
                    metrics::counter!(
                        "chronicle_pipeline_operator_emit_count",
                        "operator" => "filter",
                    )
                    .increment(1);
                    self.pending.push(FilterOut::Kept(region));
                }
                Classification::Drop(reason, tag) => {
                    metrics::counter!(
                        "chronicle_pipeline_operator_dropped_total",
                        "operator" => "filter",
                        "reason" => tag,
                    )
                    .increment(1);
                    self.pending.push(FilterOut::Dropped(DroppedRecord {
                        source_operator: "filter".into(),
                        reason,
                        details: serde_json::json!({
                            "tag": tag,
                            "text": region.transcription.text,
                            "confidence": region.transcription.confidence,
                            "start_ms": region.start_ms,
                            "end_ms": region.end_ms,
                        }),
                    }));
                }
            },
        }
        Ok(())
    }

    fn emit(&mut self) -> Vec<FilterOut> {
        std::mem::take(&mut self.pending)
    }

    async fn finalize(&mut self) -> Result<Vec<FilterOut>, PipelineError> {
        Ok(std::mem::take(&mut self.pending))
    }

    fn name(&self) -> &'static str {
        "filter"
    }
}

/// All words identical, length >= 3.
fn is_repeated_phrase(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 3 {
        return false;
    }
    let first = words[0].to_ascii_lowercase();
    words
        .iter()
        .all(|w| w.trim_end_matches(|c: char| !c.is_alphanumeric()).eq_ignore_ascii_case(&first))
}

/// Substring match against a short, closed list of known Whisper hallucinations.
fn is_known_hallucination(text: &str) -> bool {
    let normalized = text.trim().to_ascii_lowercase();
    const KNOWN: &[&str] = &[
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "subscribe to my channel",
        "www.",
        ".com",
        "[music]",
        "[applause]",
        "[laughter]",
    ];
    KNOWN.iter().any(|p| normalized.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Transcription;
    use std::sync::Arc;
    use uuid::Uuid;

    fn region(text: &str, conf: f32, dur_ms: u64) -> TranscribedRegion {
        TranscribedRegion {
            session_id: Uuid::new_v4(),
            pseudo_id: "spk".into(),
            start_ms: 0,
            end_ms: dur_ms,
            transcription: Transcription {
                text: text.into(),
                confidence: conf,
                language: None,
            },
        }
    }

    #[tokio::test]
    async fn drops_known_hallucination_thank_you_for_watching() {
        let mut op = FilterOperator::new(FilterConfig::default());
        op.ingest(TranscriptionOut::Ok(region("Thank you for watching.", -0.2, 2000)))
            .await
            .unwrap();
        let out = op.emit();
        assert!(matches!(out[0], FilterOut::Dropped(_)));
    }

    #[tokio::test]
    async fn drops_repeated_phrase() {
        let mut op = FilterOperator::new(FilterConfig::default());
        op.ingest(TranscriptionOut::Ok(region("yeah yeah yeah yeah", -0.2, 2000)))
            .await
            .unwrap();
        let out = op.emit();
        assert!(matches!(out[0], FilterOut::Dropped(_)));
    }

    #[tokio::test]
    async fn keeps_clean_segment() {
        let mut op = FilterOperator::new(FilterConfig::default());
        op.ingest(TranscriptionOut::Ok(region(
            "I draw my sword and step toward the goblin.",
            -0.2,
            3000,
        )))
        .await
        .unwrap();
        let out = op.emit();
        assert!(matches!(out[0], FilterOut::Kept(_)));
    }

    #[tokio::test]
    async fn drops_low_confidence() {
        let mut op = FilterOperator::new(FilterConfig {
            min_confidence: -0.5,
            ..Default::default()
        });
        op.ingest(TranscriptionOut::Ok(region("Hello there", -2.0, 1000)))
            .await
            .unwrap();
        let out = op.emit();
        match &out[0] {
            FilterOut::Dropped(d) => assert_eq!(d.reason, DropReason::NoiseFilter),
            other => panic!("expected dropped, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn passes_through_upstream_drop() {
        let mut op = FilterOperator::new(FilterConfig::default());
        op.ingest(TranscriptionOut::Dropped(DroppedRecord {
            source_operator: "transcription".into(),
            reason: DropReason::WhisperExhaustedRetries,
            details: serde_json::json!({}),
        }))
        .await
        .unwrap();
        let out = op.emit();
        match &out[0] {
            FilterOut::Dropped(d) => assert_eq!(d.source_operator, "transcription"),
            _ => panic!("expected dropped"),
        }
    }

    // Ensures the internal `is_repeated_phrase` works on simple patterns.
    #[test]
    fn repeated_phrase_unit() {
        assert!(is_repeated_phrase("yes yes yes"));
        assert!(!is_repeated_phrase("yes no yes"));
        assert!(!is_repeated_phrase("hi there"));
    }

    #[test]
    fn _keep_arc_alive() {
        let _a: Arc<str> = Arc::from("");
    }
}
