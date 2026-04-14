//! Meta-talk classifier.
//!
//! Tags each segment with `in_character` / `out_of_character` / `mixed`
//! / `unclear` via cheap keyword heuristics, writes the label into
//! `segment.flags.meta_talk`, and forwards the segment unchanged.

use async_trait::async_trait;

use crate::config::MetaTalkConfig;
use crate::error::PipelineError;
use crate::operator::Operator;
use crate::operators::segment::SegmentOut;
use crate::types::MetaTalkLabel;

pub struct MetaTalkOperator {
    _config: MetaTalkConfig,
    pending: Vec<SegmentOut>,
}

impl MetaTalkOperator {
    pub fn new(config: MetaTalkConfig) -> Self {
        Self {
            _config: config,
            pending: Vec::new(),
        }
    }

    fn classify(text: &str) -> MetaTalkLabel {
        let lower = text.to_ascii_lowercase();
        let strong = STRONG_OOC.iter().any(|p| lower.contains(p));
        let weak_hits = WEAK_OOC
            .iter()
            .filter(|k| word_contains(&lower, k))
            .count();

        match (strong, weak_hits) {
            (true, _) if weak_hits == 0 => MetaTalkLabel::OutOfCharacter,
            (true, _) => MetaTalkLabel::Mixed,
            (false, n) if n >= 2 => MetaTalkLabel::OutOfCharacter,
            (false, 1) => MetaTalkLabel::Unclear,
            (false, _) => MetaTalkLabel::InCharacter,
        }
    }
}

fn word_contains(haystack: &str, needle: &str) -> bool {
    if needle.contains(' ') {
        return haystack.contains(needle);
    }
    for (idx, _) in haystack.match_indices(needle) {
        let before_ok = idx == 0
            || !haystack.as_bytes()[idx - 1].is_ascii_alphanumeric();
        let after = idx + needle.len();
        let after_ok = after >= haystack.len()
            || !haystack.as_bytes()[after].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

const STRONG_OOC: &[&str] = &[
    "roll a d",
    "saving throw",
    "bonus action",
    "spell slot",
    "does that hit",
    "whose turn",
    "nat 20",
    "nat 1",
    "natural 20",
    "natural 1",
    "death save",
    "short rest",
    "long rest",
    "ability check",
    "skill check",
    "concentration check",
];

const WEAK_OOC: &[&str] = &[
    "roll", "dice", "modifier", "initiative", "advantage", "disadvantage",
    "proficiency", "d20", "d12", "d10", "d8", "d6", "d4",
];

#[async_trait]
impl Operator for MetaTalkOperator {
    type Input = SegmentOut;
    type Output = SegmentOut;

    async fn ingest(&mut self, input: SegmentOut) -> Result<(), PipelineError> {
        let span = tracing::info_span!("operator", name = "meta_talk");
        let _enter = span.enter();

        match input {
            SegmentOut::Segment(mut s) => {
                let label = Self::classify(&s.text);
                s.flags.meta_talk = Some(label);
                metrics::counter!(
                    "chronicle_pipeline_operator_emit_count",
                    "operator" => "meta_talk",
                )
                .increment(1);
                self.pending.push(SegmentOut::Segment(s));
            }
            other => self.pending.push(other),
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
        "meta_talk"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Segment, SegmentFlags};
    use uuid::Uuid;

    fn seg(text: &str) -> SegmentOut {
        SegmentOut::Segment(Segment {
            id: Uuid::now_v7(),
            session_id: Uuid::new_v4(),
            pseudo_id: "spk".into(),
            start_ms: 0,
            end_ms: 1000,
            text: text.into(),
            original: text.into(),
            confidence: -0.1,
            language: None,
            flags: SegmentFlags::default(),
        })
    }

    fn label_of(out: &SegmentOut) -> Option<MetaTalkLabel> {
        match out {
            SegmentOut::Segment(s) => s.flags.meta_talk,
            _ => None,
        }
    }

    #[tokio::test]
    async fn strong_ooc_roll_a_d20() {
        let mut op = MetaTalkOperator::new(MetaTalkConfig::default());
        op.ingest(seg("I need to roll a d20 here.")).await.unwrap();
        let out = op.emit();
        assert_eq!(label_of(&out[0]), Some(MetaTalkLabel::OutOfCharacter));
    }

    #[tokio::test]
    async fn in_character_default() {
        let mut op = MetaTalkOperator::new(MetaTalkConfig::default());
        op.ingest(seg("Greetings traveler, what brings you to our village?"))
            .await
            .unwrap();
        let out = op.emit();
        assert_eq!(label_of(&out[0]), Some(MetaTalkLabel::InCharacter));
    }

    #[tokio::test]
    async fn unclear_single_weak() {
        let mut op = MetaTalkOperator::new(MetaTalkConfig::default());
        op.ingest(seg("Sorry, what was that roll?")).await.unwrap();
        let out = op.emit();
        assert_eq!(label_of(&out[0]), Some(MetaTalkLabel::Unclear));
    }
}
