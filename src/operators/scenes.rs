//! Scene chunker operator.
//!
//! Consumes [`BeatsOut`], groups segments into `Scene`s separated by
//! `scene_break` beats or silence gaps longer than the configured
//! threshold. Emits segments + drops + beats downstream and a `Scene`
//! whenever one closes.
//!
//! Terminal operator — its `Output` is the aggregated `PipelineStep`
//! variant that the pipeline flattens into `PipelineOutput`.

use async_trait::async_trait;
use uuid::Uuid;

use crate::config::ScenesConfig;
use crate::error::PipelineError;
use crate::operator::Operator;
use crate::operators::beats::BeatsOut;
use crate::types::{Beat, BeatKind, DroppedRecord, Scene, Segment, SessionId};

/// Terminal output variant flattened by the pipeline into
/// [`crate::types::PipelineOutput`].
#[derive(Debug, Clone)]
pub enum ScenesOut {
    Segment(Segment),
    Beat(Beat),
    Scene(Scene),
    Dropped(DroppedRecord),
}

pub struct ScenesOperator {
    config: ScenesConfig,
    pending: Vec<ScenesOut>,
    state: SceneState,
}

#[derive(Debug, Clone)]
enum SceneState {
    Empty,
    Open {
        session_id: SessionId,
        start_ms: u64,
        last_end_ms: u64,
        segment_count: u32,
    },
}

impl ScenesOperator {
    pub fn new(config: ScenesConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            state: SceneState::Empty,
        }
    }

    fn close_scene_if_open(&mut self, label: &str) -> Option<Scene> {
        match std::mem::replace(&mut self.state, SceneState::Empty) {
            SceneState::Empty => None,
            SceneState::Open {
                session_id,
                start_ms,
                last_end_ms,
                segment_count,
            } if segment_count > 0 => Some(Scene {
                id: Uuid::now_v7(),
                session_id,
                start_ms,
                end_ms: last_end_ms,
                label: label.into(),
                confidence: 0.5,
            }),
            SceneState::Open { .. } => None,
        }
    }

    fn handle_segment(&mut self, seg: &Segment) {
        let gap_triggers_split = matches!(
            self.state,
            SceneState::Open { last_end_ms, .. }
                if seg.start_ms.saturating_sub(last_end_ms) >= self.config.max_silence_gap_ms,
        );
        let duration_triggers_split = matches!(
            self.state,
            SceneState::Open { start_ms, .. }
                if seg.end_ms.saturating_sub(start_ms) >= self.config.max_scene_ms,
        );

        if gap_triggers_split || duration_triggers_split {
            let label = if gap_triggers_split {
                "silence_gap"
            } else {
                "max_duration"
            };
            if let Some(scene) = self.close_scene_if_open(label) {
                self.emit_scene(scene);
            }
        }

        match &mut self.state {
            SceneState::Empty => {
                self.state = SceneState::Open {
                    session_id: seg.session_id,
                    start_ms: seg.start_ms,
                    last_end_ms: seg.end_ms,
                    segment_count: 1,
                };
            }
            SceneState::Open {
                last_end_ms,
                segment_count,
                ..
            } => {
                *last_end_ms = (*last_end_ms).max(seg.end_ms);
                *segment_count += 1;
            }
        }
    }

    fn handle_beat(&mut self, beat: &Beat) {
        if beat.kind == BeatKind::SceneBreak {
            if let Some(scene) = self.close_scene_if_open("scene_break") {
                self.emit_scene(scene);
            }
        }
    }

    fn emit_scene(&mut self, scene: Scene) {
        metrics::counter!(
            "chronicle_pipeline_operator_emit_count",
            "operator" => "scenes",
        )
        .increment(1);
        self.pending.push(ScenesOut::Scene(scene));
    }
}

#[async_trait]
impl Operator for ScenesOperator {
    type Input = BeatsOut;
    type Output = ScenesOut;

    async fn ingest(&mut self, input: BeatsOut) -> Result<(), PipelineError> {
        let span = tracing::info_span!("operator", name = "scenes");
        let _enter = span.enter();

        match input {
            BeatsOut::Segment(s) => {
                self.handle_segment(&s);
                self.pending.push(ScenesOut::Segment(s));
            }
            BeatsOut::Beat(b) => {
                self.handle_beat(&b);
                self.pending.push(ScenesOut::Beat(b));
            }
            BeatsOut::Dropped(d) => {
                self.pending.push(ScenesOut::Dropped(d));
            }
        }
        Ok(())
    }

    fn emit(&mut self) -> Vec<ScenesOut> {
        std::mem::take(&mut self.pending)
    }

    async fn finalize(&mut self) -> Result<Vec<ScenesOut>, PipelineError> {
        if let Some(scene) = self.close_scene_if_open("end_of_session") {
            self.emit_scene(scene);
        }
        Ok(std::mem::take(&mut self.pending))
    }

    fn name(&self) -> &'static str {
        "scenes"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SegmentFlags;
    use uuid::Uuid;

    fn make_seg(session: Uuid, start: u64, end: u64) -> Segment {
        Segment {
            id: Uuid::now_v7(),
            session_id: session,
            pseudo_id: "spk".into(),
            start_ms: start,
            end_ms: end,
            text: "x".into(),
            original: "x".into(),
            confidence: -0.1,
            language: None,
            flags: SegmentFlags::default(),
        }
    }

    #[tokio::test]
    async fn splits_on_silence_gap() {
        let session = Uuid::new_v4();
        let mut op = ScenesOperator::new(ScenesConfig {
            max_silence_gap_ms: 5_000,
            max_scene_ms: 600_000,
        });
        op.ingest(BeatsOut::Segment(make_seg(session, 0, 1000))).await.unwrap();
        op.ingest(BeatsOut::Segment(make_seg(session, 10_000, 11_000)))
            .await
            .unwrap();
        let out = op.finalize().await.unwrap();
        let scenes: Vec<_> = out
            .iter()
            .filter_map(|o| match o {
                ScenesOut::Scene(s) => Some(s),
                _ => None,
            })
            .collect();
        assert_eq!(scenes.len(), 2, "expected two scenes, got {out:?}");
    }

    #[tokio::test]
    async fn closes_scene_on_scene_break_beat() {
        let session = Uuid::new_v4();
        let mut op = ScenesOperator::new(ScenesConfig::default());
        op.ingest(BeatsOut::Segment(make_seg(session, 0, 1000))).await.unwrap();
        op.ingest(BeatsOut::Beat(Beat {
            id: Uuid::now_v7(),
            session_id: session,
            t_ms: 1000,
            kind: BeatKind::SceneBreak,
            label: "scene_break".into(),
            confidence: 0.5,
        }))
        .await
        .unwrap();
        op.ingest(BeatsOut::Segment(make_seg(session, 2000, 3000))).await.unwrap();
        let out = op.finalize().await.unwrap();
        let scenes: Vec<_> = out
            .iter()
            .filter_map(|o| match o {
                ScenesOut::Scene(s) => Some(s),
                _ => None,
            })
            .collect();
        assert_eq!(scenes.len(), 2);
    }
}
