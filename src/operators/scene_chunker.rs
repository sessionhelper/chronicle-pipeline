//! Scene operator: assigns `chunk_group` to segments based on silence
//! gaps and duration limits.
//!
//! A new scene starts when either:
//! - The silence gap between consecutive segments exceeds `max_silence_gap`
//! - The current scene's duration exceeds `max_chunk_duration`

use crate::error::Result;
use crate::types::TranscriptSegment;

use super::{OperatorResult, Operator};

/// Configuration for scene boundary detection.
#[derive(Debug, Clone)]
pub struct SceneOperatorConfig {
    /// Maximum silence gap in seconds before starting a new scene.
    pub max_silence_gap: f32,
    /// Maximum duration of a single scene in seconds.
    pub max_chunk_duration: f32,
}

impl Default for SceneOperatorConfig {
    fn default() -> Self {
        Self {
            max_silence_gap: 30.0,
            max_chunk_duration: 600.0,
        }
    }
}

/// Scene operator. Assigns sequential `chunk_group` numbers to
/// transcript segments for UI organization.
pub struct SceneOperator {
    config: SceneOperatorConfig,
    /// Current scene index (0-based).
    current_group: u32,
    /// Start time of the current scene.
    scene_start: Option<f32>,
    /// End time of the most recent segment in the current scene.
    last_end: Option<f32>,
    /// Total number of scenes created.
    scenes_created: u32,
}

impl SceneOperator {
    /// Create a new scene operator with the given configuration.
    pub fn new(config: SceneOperatorConfig) -> Self {
        Self {
            config,
            current_group: 0,
            scene_start: None,
            last_end: None,
            scenes_created: 0,
        }
    }

    /// Check whether we should start a new scene based on the gap since
    /// the last segment and the current scene's duration.
    fn should_split(&self, segment_start: f32) -> bool {
        let Some(last_end) = self.last_end else {
            return false; // First segment, no split needed
        };
        let Some(scene_start) = self.scene_start else {
            return false;
        };

        // Split on silence gap
        let gap = segment_start - last_end;
        if gap >= self.config.max_silence_gap {
            return true;
        }

        // Split on scene duration
        let scene_duration = segment_start - scene_start;
        if scene_duration >= self.config.max_chunk_duration {
            return true;
        }

        false
    }
}

#[async_trait::async_trait]
impl Operator for SceneOperator {
    async fn on_segment(&mut self, segment: &mut TranscriptSegment) -> OperatorResult {
        // Skip excluded segments — they don't affect scene boundaries
        if segment.excluded {
            return OperatorResult::Pass;
        }

        if self.scene_start.is_none() {
            // First segment starts the first scene
            self.scene_start = Some(segment.start_time);
            self.scenes_created = 1;
        } else if self.should_split(segment.start_time) {
            // Start a new scene
            self.current_group += 1;
            self.scene_start = Some(segment.start_time);
            self.scenes_created += 1;

            tracing::debug!(
                group = self.current_group,
                start = segment.start_time,
                "new scene boundary"
            );
        }

        segment.chunk_group = Some(self.current_group);
        self.last_end = Some(segment.end_time);

        OperatorResult::Pass
    }

    async fn sweep(&mut self) -> Result<u32> {
        // Scene operator doesn't do retroactive analysis
        Ok(0)
    }

    async fn finalize(&mut self) -> Result<()> {
        tracing::info!(
            scenes = self.scenes_created,
            "scene operator finalized"
        );
        Ok(())
    }
}

impl SceneOperator {
    /// Return the total number of scenes detected so far.
    pub fn scenes_detected(&self) -> u32 {
        self.scenes_created
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_segment(start: f32, end: f32) -> TranscriptSegment {
        TranscriptSegment {
            id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            segment_index: 0,
            speaker_pseudo_id: "speaker_a".into(),
            start_time: start,
            end_time: end,
            text: "test".into(),
            original_text: "test".into(),
            confidence: None,
            chunk_group: None,
            excluded: false,
            exclude_reason: None,
        }
    }

    #[tokio::test]
    async fn assigns_same_group_for_close_segments() {
        let mut chunker = SceneOperator::new(SceneOperatorConfig::default());

        let mut s1 = make_segment(0.0, 5.0);
        let mut s2 = make_segment(6.0, 10.0);
        let mut s3 = make_segment(11.0, 15.0);

        chunker.on_segment(&mut s1).await;
        chunker.on_segment(&mut s2).await;
        chunker.on_segment(&mut s3).await;

        assert_eq!(s1.chunk_group, Some(0));
        assert_eq!(s2.chunk_group, Some(0));
        assert_eq!(s3.chunk_group, Some(0));
    }

    #[tokio::test]
    async fn splits_on_silence_gap() {
        let config = SceneOperatorConfig {
            max_silence_gap: 10.0,
            max_chunk_duration: 600.0,
        };
        let mut chunker = SceneOperator::new(config);

        let mut s1 = make_segment(0.0, 5.0);
        let mut s2 = make_segment(20.0, 25.0); // 15s gap > 10s threshold

        chunker.on_segment(&mut s1).await;
        chunker.on_segment(&mut s2).await;

        assert_eq!(s1.chunk_group, Some(0));
        assert_eq!(s2.chunk_group, Some(1));
        assert_eq!(chunker.scenes_detected(), 2);
    }

    #[tokio::test]
    async fn splits_on_duration_limit() {
        let config = SceneOperatorConfig {
            max_silence_gap: 30.0,
            max_chunk_duration: 100.0,
        };
        let mut chunker = SceneOperator::new(config);

        let mut s1 = make_segment(0.0, 50.0);
        let mut s2 = make_segment(51.0, 90.0);
        let mut s3 = make_segment(101.0, 110.0); // scene duration > 100s

        chunker.on_segment(&mut s1).await;
        chunker.on_segment(&mut s2).await;
        chunker.on_segment(&mut s3).await;

        assert_eq!(s1.chunk_group, Some(0));
        assert_eq!(s2.chunk_group, Some(0));
        assert_eq!(s3.chunk_group, Some(1));
    }
}
