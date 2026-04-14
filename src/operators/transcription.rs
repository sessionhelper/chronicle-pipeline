//! Transcription operator.
//!
//! Consumes [`VoiceRegion`]s, calls the caller-supplied
//! [`WhisperClient`], and emits [`TranscribedRegion`]s. Implements the
//! 3-attempt exponential backoff policy specced at 500 ms initial.
//!
//! Per-input failures (Whisper exhausted retries, bad payload) do not
//! bubble up — they become `DroppedRecord` via the `Output` enum.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::config::TranscriptionConfig;
use crate::error::{PipelineError, WhisperError};
use crate::operator::Operator;
use crate::types::{DropReason, DroppedRecord, TranscribedRegion, Transcription, VoiceRegion};
use crate::whisper::WhisperClient;

/// Output of the transcription operator. Either a successful
/// transcription or a drop record.
#[derive(Debug, Clone)]
pub enum TranscriptionOut {
    Ok(TranscribedRegion),
    Dropped(DroppedRecord),
}

pub struct TranscriptionOperator {
    config: TranscriptionConfig,
    whisper: Arc<dyn WhisperClient>,
    pending: Vec<TranscriptionOut>,
}

impl TranscriptionOperator {
    pub fn new(config: TranscriptionConfig, whisper: Arc<dyn WhisperClient>) -> Self {
        Self {
            config,
            whisper,
            pending: Vec::new(),
        }
    }

    async fn transcribe_with_retry(
        &self,
        region: &VoiceRegion,
    ) -> Result<Transcription, WhisperError> {
        let mut backoff = Duration::from_millis(self.config.initial_backoff_ms);
        let mut last_err: Option<WhisperError> = None;

        for attempt in 0..self.config.max_attempts {
            let span = tracing::info_span!(
                "whisper_call",
                region_ms = region.end_ms.saturating_sub(region.start_ms),
                bytes = region.pcm.len() * std::mem::size_of::<f32>(),
                attempt,
            );
            let _enter = span.enter();

            let start = std::time::Instant::now();
            let res = self.whisper.transcribe(&region.pcm, 16_000).await;
            let latency_ms = start.elapsed().as_millis() as f64;
            metrics::histogram!("chronicle_pipeline_whisper_latency_ms").record(latency_ms);

            match res {
                Ok(t) => return Ok(t),
                Err(err) if err.is_transient() && attempt + 1 < self.config.max_attempts => {
                    metrics::counter!("chronicle_pipeline_whisper_retries_total").increment(1);
                    tracing::warn!(
                        attempt,
                        backoff_ms = backoff.as_millis() as u64,
                        error = %err,
                        "whisper transient failure, retrying"
                    );
                    tokio::time::sleep(backoff).await;
                    backoff *= 2;
                    last_err = Some(err);
                }
                Err(err) => {
                    return Err(err);
                }
            }
        }
        Err(last_err.unwrap_or(WhisperError::Fatal("no attempts".into())))
    }
}

#[async_trait]
impl Operator for TranscriptionOperator {
    type Input = VoiceRegion;
    type Output = TranscriptionOut;

    async fn ingest(&mut self, input: VoiceRegion) -> Result<(), PipelineError> {
        let span = tracing::info_span!("operator", name = "transcription");
        let _enter = span.enter();

        if input.pcm.is_empty() {
            self.pending.push(TranscriptionOut::Dropped(DroppedRecord {
                source_operator: "transcription".into(),
                reason: DropReason::InvalidVadRegion,
                details: serde_json::json!({
                    "start_ms": input.start_ms,
                    "end_ms": input.end_ms,
                }),
            }));
            return Ok(());
        }

        match self.transcribe_with_retry(&input).await {
            Ok(t) => {
                self.pending.push(TranscriptionOut::Ok(TranscribedRegion {
                    session_id: input.session_id,
                    pseudo_id: input.pseudo_id,
                    start_ms: input.start_ms,
                    end_ms: input.end_ms,
                    transcription: t,
                }));
                metrics::counter!(
                    "chronicle_pipeline_operator_emit_count",
                    "operator" => "transcription",
                )
                .increment(1);
            }
            Err(err) => {
                let reason = match err {
                    WhisperError::Transient(_) => DropReason::WhisperExhaustedRetries,
                    WhisperError::Fatal(_) => DropReason::WhisperBadPayload,
                };
                metrics::counter!(
                    "chronicle_pipeline_operator_dropped_total",
                    "operator" => "transcription",
                    "reason" => reason_label(reason),
                )
                .increment(1);
                self.pending.push(TranscriptionOut::Dropped(DroppedRecord {
                    source_operator: "transcription".into(),
                    reason,
                    details: serde_json::json!({
                        "start_ms": input.start_ms,
                        "end_ms": input.end_ms,
                        "error": err.to_string(),
                    }),
                }));
            }
        }
        Ok(())
    }

    fn emit(&mut self) -> Vec<TranscriptionOut> {
        std::mem::take(&mut self.pending)
    }

    async fn finalize(&mut self) -> Result<Vec<TranscriptionOut>, PipelineError> {
        Ok(std::mem::take(&mut self.pending))
    }

    fn name(&self) -> &'static str {
        "transcription"
    }
}

fn reason_label(reason: DropReason) -> &'static str {
    match reason {
        DropReason::InvalidVadRegion => "invalid_vad_region",
        DropReason::WhisperExhaustedRetries => "whisper_exhausted_retries",
        DropReason::WhisperBadPayload => "whisper_bad_payload",
        DropReason::Hallucination => "hallucination",
        DropReason::NoiseFilter => "noise_filter",
        DropReason::HeuristicReject => "heuristic_reject",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use uuid::Uuid;

    struct MockWhisper {
        fail_first: u32,
        calls: AtomicU32,
        fatal: bool,
    }

    #[async_trait]
    impl WhisperClient for MockWhisper {
        async fn transcribe(
            &self,
            _audio: &[f32],
            _sr: u32,
        ) -> Result<Transcription, WhisperError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_first {
                if self.fatal {
                    return Err(WhisperError::Fatal("boom".into()));
                }
                return Err(WhisperError::Transient("retry me".into()));
            }
            Ok(Transcription {
                text: "hello".into(),
                confidence: -0.1,
                language: Some("en".into()),
            })
        }
    }

    fn region() -> VoiceRegion {
        VoiceRegion {
            session_id: Uuid::new_v4(),
            pseudo_id: "spk".into(),
            start_ms: 0,
            end_ms: 500,
            pcm: Arc::from(vec![0.1; 8000]),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn succeeds_after_transient_failures() {
        let whisper = Arc::new(MockWhisper {
            fail_first: 2,
            calls: AtomicU32::new(0),
            fatal: false,
        });
        let mut op = TranscriptionOperator::new(
            TranscriptionConfig {
                max_attempts: 3,
                initial_backoff_ms: 500,
            },
            whisper.clone(),
        );
        op.ingest(region()).await.unwrap();
        let out = op.finalize().await.unwrap();
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], TranscriptionOut::Ok(_)));
        assert_eq!(whisper.calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn drops_after_exhausted_retries() {
        let whisper = Arc::new(MockWhisper {
            fail_first: 10,
            calls: AtomicU32::new(0),
            fatal: false,
        });
        let mut op = TranscriptionOperator::new(
            TranscriptionConfig {
                max_attempts: 3,
                initial_backoff_ms: 500,
            },
            whisper.clone(),
        );
        op.ingest(region()).await.unwrap();
        let out = op.finalize().await.unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            TranscriptionOut::Dropped(d) => {
                assert_eq!(d.reason, DropReason::WhisperExhaustedRetries);
            }
            other => panic!("expected dropped, got {other:?}"),
        }
        assert_eq!(whisper.calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn fatal_skips_retry_and_drops_immediately() {
        let whisper = Arc::new(MockWhisper {
            fail_first: 10,
            calls: AtomicU32::new(0),
            fatal: true,
        });
        let mut op = TranscriptionOperator::new(
            TranscriptionConfig {
                max_attempts: 3,
                initial_backoff_ms: 500,
            },
            whisper.clone(),
        );
        op.ingest(region()).await.unwrap();
        let out = op.finalize().await.unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            TranscriptionOut::Dropped(d) => {
                assert_eq!(d.reason, DropReason::WhisperBadPayload);
            }
            other => panic!("expected dropped, got {other:?}"),
        }
        assert_eq!(whisper.calls.load(Ordering::SeqCst), 1);
    }
}
