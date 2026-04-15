//! Beat detection operator.
//!
//! Consumes [`SegmentOut`] (segments + passthrough drops), emits
//! [`BeatsOut`] which carries forwarded segments/drops and newly-detected
//! beats. The detector is a pure heuristic state machine over the
//! segment stream: silence gaps trigger scene breaks, keyword hits
//! trigger combat / discovery / dialogue-climax beats.
//!
//! Upgrading to an LLM-backed detector is a matter of swapping the
//! classifier — the operator shape stays the same.

use async_trait::async_trait;
use uuid::Uuid;

use crate::config::BeatsConfig;
use crate::error::PipelineError;
use crate::operator::Operator;
use crate::operators::segment::SegmentOut;
use crate::types::{Beat, BeatKind, DroppedRecord, Segment, SessionId};

/// Output carrying either a segment/drop forwarded downstream or a
/// newly-emitted beat.
#[derive(Debug, Clone)]
pub enum BeatsOut {
    Segment(Segment),
    Dropped(DroppedRecord),
    Beat(Beat),
}

pub struct BeatsOperator {
    config: BeatsConfig,
    pending: Vec<BeatsOut>,
    state: BeatState,
}

/// Beat-detection FSM state. No open-ended flags — everything is a
/// variant.
#[derive(Debug, Clone)]
enum BeatState {
    /// No prior segment yet.
    Fresh,
    /// We've seen at least one segment; tracking gap to next.
    Tracking {
        session_id: SessionId,
        last_end_ms: u64,
        in_combat: bool,
    },
}

impl BeatsOperator {
    pub fn new(config: BeatsConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            state: BeatState::Fresh,
        }
    }

    fn push_beat(&mut self, session_id: SessionId, t_ms: u64, kind: BeatKind, label: &str) {
        metrics::counter!(
            "chronicle_pipeline_operator_emit_count",
            "operator" => "beats",
        )
        .increment(1);
        self.pending.push(BeatsOut::Beat(Beat {
            id: Uuid::now_v7(),
            session_id,
            t_ms,
            kind,
            label: label.into(),
            confidence: 0.5,
        }));
    }

    fn process_segment(&mut self, seg: &Segment) {
        if seg.confidence < self.config.min_segment_confidence {
            return;
        }

        let lower = seg.text.to_ascii_lowercase();
        let (new_state, emitted_kinds) = match self.state.clone() {
            BeatState::Fresh => {
                let mut emits = Vec::new();
                let in_combat = mentions_combat_start(&lower);
                if in_combat {
                    emits.push((seg.start_ms, BeatKind::CombatStart, "combat_start"));
                }
                if mentions_discovery(&lower) {
                    emits.push((seg.start_ms, BeatKind::Discovery, "discovery"));
                }
                (
                    BeatState::Tracking {
                        session_id: seg.session_id,
                        last_end_ms: seg.end_ms,
                        in_combat,
                    },
                    emits,
                )
            }
            BeatState::Tracking {
                session_id,
                last_end_ms,
                in_combat,
            } => {
                let mut emits = Vec::new();
                let gap_ms = seg.start_ms.saturating_sub(last_end_ms);

                if gap_ms >= self.config.scene_break_silence_ms {
                    emits.push((seg.start_ms, BeatKind::SceneBreak, "scene_break"));
                }

                let mut new_in_combat = in_combat;
                if !in_combat && mentions_combat_start(&lower) {
                    emits.push((seg.start_ms, BeatKind::CombatStart, "combat_start"));
                    new_in_combat = true;
                } else if in_combat && mentions_combat_end(&lower) {
                    emits.push((seg.start_ms, BeatKind::CombatEnd, "combat_end"));
                    new_in_combat = false;
                }
                if mentions_discovery(&lower) {
                    emits.push((seg.start_ms, BeatKind::Discovery, "discovery"));
                }
                if mentions_dialogue_climax(&lower) {
                    emits.push((seg.start_ms, BeatKind::DialogueClimax, "dialogue_climax"));
                }

                (
                    BeatState::Tracking {
                        session_id,
                        last_end_ms: seg.end_ms.max(last_end_ms),
                        in_combat: new_in_combat,
                    },
                    emits,
                )
            }
        };

        self.state = new_state;
        for (t_ms, kind, label) in emitted_kinds {
            self.push_beat(seg.session_id, t_ms, kind, label);
        }
    }
}

fn mentions_combat_start(lower: &str) -> bool {
    const CUES: &[&str] = &[
        "roll initiative",
        "roll for initiative",
        "attack",
        "i attack",
        "we attack",
        "swords drawn",
    ];
    CUES.iter().any(|c| lower.contains(c))
}

fn mentions_combat_end(lower: &str) -> bool {
    const CUES: &[&str] = &[
        "it falls",
        "they fall",
        "they're dead",
        "combat over",
        "combat ends",
        "everyone sheathes",
    ];
    CUES.iter().any(|c| lower.contains(c))
}

fn mentions_discovery(lower: &str) -> bool {
    const CUES: &[&str] = &[
        "you find",
        "you discover",
        "you notice",
        "you see a",
        "hidden door",
        "secret passage",
    ];
    CUES.iter().any(|c| lower.contains(c))
}

fn mentions_dialogue_climax(lower: &str) -> bool {
    const CUES: &[&str] = &[
        "confess",
        "swore an oath",
        "betrayed",
        "reveal",
        "told the truth",
    ];
    CUES.iter().any(|c| lower.contains(c))
}

#[async_trait]
impl Operator for BeatsOperator {
    type Input = SegmentOut;
    type Output = BeatsOut;

    async fn ingest(&mut self, input: SegmentOut) -> Result<(), PipelineError> {
        let span = tracing::info_span!("operator", name = "beats");
        let _enter = span.enter();

        match input {
            SegmentOut::Segment(s) => {
                self.process_segment(&s);
                self.pending.push(BeatsOut::Segment(s));
            }
            SegmentOut::Dropped(d) => {
                self.pending.push(BeatsOut::Dropped(d));
            }
        }
        Ok(())
    }

    fn emit(&mut self) -> Vec<BeatsOut> {
        std::mem::take(&mut self.pending)
    }

    async fn finalize(&mut self) -> Result<Vec<BeatsOut>, PipelineError> {
        Ok(std::mem::take(&mut self.pending))
    }

    fn name(&self) -> &'static str {
        "beats"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SegmentFlags;
    use uuid::Uuid;

    fn seg(session: Uuid, text: &str, start: u64, end: u64) -> SegmentOut {
        SegmentOut::Segment(Segment {
            id: Uuid::now_v7(),
            session_id: session,
            pseudo_id: "spk".into(),
            start_ms: start,
            end_ms: end,
            text: text.into(),
            original: text.into(),
            confidence: -0.1,
            language: None,
            flags: SegmentFlags::default(),
        })
    }

    #[tokio::test]
    async fn detects_combat_start_and_end() {
        let session = Uuid::new_v4();
        let mut op = BeatsOperator::new(BeatsConfig::default());
        op.ingest(seg(session, "Roll initiative!", 0, 1000)).await.unwrap();
        op.ingest(seg(session, "I hit the goblin.", 1500, 2500)).await.unwrap();
        op.ingest(seg(session, "It falls to the ground.", 3000, 4000)).await.unwrap();
        let beats: Vec<_> = op
            .emit()
            .into_iter()
            .filter_map(|b| match b {
                BeatsOut::Beat(b) => Some(b),
                _ => None,
            })
            .collect();
        assert!(beats.iter().any(|b| b.kind == BeatKind::CombatStart));
        assert!(beats.iter().any(|b| b.kind == BeatKind::CombatEnd));
    }

    #[tokio::test]
    async fn detects_scene_break_on_silence() {
        let session = Uuid::new_v4();
        let mut op = BeatsOperator::new(BeatsConfig {
            scene_break_silence_ms: 5000,
            ..Default::default()
        });
        op.ingest(seg(session, "Hello there.", 0, 1000)).await.unwrap();
        op.ingest(seg(session, "Later that evening.", 10000, 11000)).await.unwrap();
        let beats: Vec<_> = op
            .emit()
            .into_iter()
            .filter_map(|b| match b {
                BeatsOut::Beat(b) => Some(b),
                _ => None,
            })
            .collect();
        assert!(beats.iter().any(|b| b.kind == BeatKind::SceneBreak));
    }
}
