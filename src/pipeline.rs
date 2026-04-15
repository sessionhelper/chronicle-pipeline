//! `Pipeline` — composes operators in the order specified by
//! [`PipelineConfig::operators`].
//!
//! The compile-time strong typing of `Operator::Input` / `Output`
//! makes a perfectly generic chain awkward, so the pipeline uses a
//! typed chain of adapters hand-wired in the builder. The caller sees
//! a single `Pipeline` with the public `ingest_chunk` / `emit` /
//! `finalize` / `run_one_shot` surface.

use std::sync::Arc;

use crate::config::{OperatorKind, PipelineConfig};
use crate::error::{PipelineError, Result};
use crate::operator::Operator;
use crate::operators::{
    beats::{BeatsOperator, BeatsOut},
    filter::{FilterOperator, FilterOut},
    meta_talk::MetaTalkOperator,
    scenes::{ScenesOperator, ScenesOut},
    segment::{SegmentOperator, SegmentOut},
    transcription::{TranscriptionOperator, TranscriptionOut},
    vad::{VadEngine, VadOperator},
};
use crate::types::{
    AudioChunk, PipelineOutput, SessionAudio, SessionId, SessionTrack, Timestamp, VoiceRegion,
};
use crate::whisper::WhisperClient;

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

pub struct PipelineBuilder {
    config: PipelineConfig,
    whisper: Option<Arc<dyn WhisperClient>>,
    vad_engine: Option<Box<dyn VadEngine>>,
}

impl PipelineBuilder {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            whisper: None,
            vad_engine: None,
        }
    }

    pub fn whisper(mut self, whisper: Arc<dyn WhisperClient>) -> Self {
        self.whisper = Some(whisper);
        self
    }

    pub fn vad_engine(mut self, engine: Box<dyn VadEngine>) -> Self {
        self.vad_engine = Some(engine);
        self
    }

    pub fn build(mut self) -> Result<Pipeline> {
        // Validate operator ordering: whatever appears must appear in
        // canonical order, with no duplicates.
        validate_operator_order(&self.config.operators)?;

        let mut enabled = [false; 7];
        for kind in &self.config.operators {
            enabled[*kind as usize] = true;
        }

        // VAD
        let vad = if enabled[OperatorKind::Vad as usize] {
            let engine = self
                .vad_engine
                .take()
                .or_else(|| default_vad_engine(&self.config))
                .ok_or_else(|| {
                    PipelineError::ConfigInvalid(
                        "vad operator enabled but no VAD engine provided and no default available"
                            .into(),
                    )
                })?;
            Some(VadOperator::new(self.config.vad.clone(), engine))
        } else {
            None
        };

        // Transcription
        let transcription = if enabled[OperatorKind::Transcription as usize] {
            let whisper = self.whisper.clone().ok_or_else(|| {
                PipelineError::ConfigInvalid(
                    "transcription operator enabled but no WhisperClient provided".into(),
                )
            })?;
            Some(TranscriptionOperator::new(
                self.config.transcription.clone(),
                whisper,
            ))
        } else {
            None
        };

        let filter = enabled[OperatorKind::Filter as usize]
            .then(|| FilterOperator::new(self.config.filter.clone()));
        let segment = enabled[OperatorKind::Segment as usize].then(SegmentOperator::new);
        let meta_talk = enabled[OperatorKind::MetaTalk as usize]
            .then(|| MetaTalkOperator::new(self.config.meta_talk.clone()));
        let beats = enabled[OperatorKind::Beats as usize]
            .then(|| BeatsOperator::new(self.config.beats.clone()));
        let scenes = enabled[OperatorKind::Scenes as usize]
            .then(|| ScenesOperator::new(self.config.scenes.clone()));

        Ok(Pipeline {
            vad,
            transcription,
            filter,
            segment,
            meta_talk,
            beats,
            scenes,
            drainable: PipelineOutput::default(),
        })
    }
}

fn validate_operator_order(ops: &[OperatorKind]) -> Result<()> {
    // Walk the canonical order and make sure `ops` is a subsequence with
    // no duplicates. Any other ordering is rejected.
    let canonical = OperatorKind::ALL;
    let mut cursor = 0;
    for op in ops {
        let pos = canonical[cursor..]
            .iter()
            .position(|k| k == op)
            .ok_or_else(|| {
                PipelineError::ConfigInvalid(format!(
                    "operator {:?} appears out of canonical order or duplicated",
                    op
                ))
            })?;
        cursor += pos + 1;
    }
    Ok(())
}

#[cfg(feature = "vad")]
fn default_vad_engine(config: &PipelineConfig) -> Option<Box<dyn VadEngine>> {
    use crate::operators::vad::silero::SileroEngine;
    config
        .vad
        .model_path
        .as_ref()
        .map(|path| -> Box<dyn VadEngine> { Box::new(SileroEngine::new(path.clone())) })
}

#[cfg(not(feature = "vad"))]
fn default_vad_engine(_config: &PipelineConfig) -> Option<Box<dyn VadEngine>> {
    None
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

pub struct Pipeline {
    vad: Option<VadOperator>,
    transcription: Option<TranscriptionOperator>,
    filter: Option<FilterOperator>,
    segment: Option<SegmentOperator>,
    meta_talk: Option<MetaTalkOperator>,
    beats: Option<BeatsOperator>,
    scenes: Option<ScenesOperator>,
    drainable: PipelineOutput,
}

impl Pipeline {
    pub fn builder(config: PipelineConfig) -> PipelineBuilder {
        PipelineBuilder::new(config)
    }

    /// Streaming ingest. Pushes a single audio chunk into the pipeline.
    /// Downstream operators run synchronously to completion for this
    /// chunk's contribution — anything ready to emit is stashed in
    /// `drainable` for the next `emit()` call.
    pub async fn ingest_chunk(&mut self, chunk: AudioChunk) -> Result<()> {
        let session_id = chunk.session_id;

        // VAD
        let regions = if let Some(vad) = self.vad.as_mut() {
            vad.ingest(chunk).await?;
            vad.emit()
        } else {
            return Err(PipelineError::ConfigInvalid(
                "pipeline has no VAD operator but ingest_chunk was called".into(),
            ));
        };

        self.process_regions(session_id, regions).await
    }

    /// One-shot: feed a whole session, return the complete output.
    pub async fn run_one_shot(mut self, audio: SessionAudio) -> Result<PipelineOutput> {
        let session_id = audio.session_id;
        for track in audio.tracks {
            let chunks = split_track_into_chunks(session_id, track);
            for chunk in chunks {
                self.ingest_chunk(chunk).await?;
            }
        }
        self.finalize().await
    }

    /// Drain whatever output is ready for the caller. Does not perform
    /// any ingest work itself.
    pub fn emit(&mut self) -> PipelineOutput {
        let mut out = std::mem::take(&mut self.drainable);
        out.sort_in_place();
        out
    }

    /// End-of-input. Flushes every operator and returns the complete
    /// remaining output.
    pub async fn finalize(mut self) -> Result<PipelineOutput> {
        // Drain remaining VAD regions.
        let regions = if let Some(vad) = self.vad.as_mut() {
            vad.finalize().await?
        } else {
            Vec::new()
        };

        let session_id = regions.first().map(|r| r.session_id).unwrap_or_default();
        self.process_regions(session_id, regions).await?;

        // Finalize downstream chain in order: each op's finalize may
        // emit more, which we propagate.
        if let Some(t) = self.transcription.as_mut() {
            let out = t.finalize().await?;
            self.forward_transcription(out).await?;
        }
        if let Some(f) = self.filter.as_mut() {
            let out = f.finalize().await?;
            self.forward_filter(out).await?;
        }
        if let Some(s) = self.segment.as_mut() {
            let out = s.finalize().await?;
            self.forward_segment(out).await?;
        }
        if let Some(m) = self.meta_talk.as_mut() {
            let out = m.finalize().await?;
            self.forward_meta_talk(out).await?;
        }
        if let Some(b) = self.beats.as_mut() {
            let out = b.finalize().await?;
            self.forward_beats(out).await?;
        }
        if let Some(s) = self.scenes.as_mut() {
            let out = s.finalize().await?;
            self.absorb_scenes_out(out);
        }

        let mut final_out = std::mem::take(&mut self.drainable);
        final_out.sort_in_place();
        Ok(final_out)
    }

    // ---- Chain helpers ----

    async fn process_regions(
        &mut self,
        _session_id: SessionId,
        regions: Vec<VoiceRegion>,
    ) -> Result<()> {
        if let Some(t) = self.transcription.as_mut() {
            for region in regions {
                t.ingest(region).await?;
            }
            let out = t.emit();
            self.forward_transcription(out).await?;
        }
        Ok(())
    }

    async fn forward_transcription(&mut self, out: Vec<TranscriptionOut>) -> Result<()> {
        if let Some(f) = self.filter.as_mut() {
            for item in out {
                f.ingest(item).await?;
            }
            let emitted = f.emit();
            self.forward_filter(emitted).await?;
        } else {
            // No filter — synthesize drops/segments directly? Spec says
            // filter is mandatory. Drop silently otherwise.
        }
        Ok(())
    }

    async fn forward_filter(&mut self, out: Vec<FilterOut>) -> Result<()> {
        if let Some(s) = self.segment.as_mut() {
            for item in out {
                s.ingest(item).await?;
            }
            let emitted = s.emit();
            self.forward_segment(emitted).await?;
        }
        Ok(())
    }

    async fn forward_segment(&mut self, out: Vec<SegmentOut>) -> Result<()> {
        if let Some(m) = self.meta_talk.as_mut() {
            for item in out {
                m.ingest(item).await?;
            }
            let emitted = m.emit();
            self.forward_meta_talk(emitted).await?;
        } else {
            // Skip meta-talk; forward direct to beats if present.
            self.forward_meta_talk(out).await?;
        }
        Ok(())
    }

    async fn forward_meta_talk(&mut self, out: Vec<SegmentOut>) -> Result<()> {
        if let Some(b) = self.beats.as_mut() {
            for item in out {
                b.ingest(item).await?;
            }
            let emitted = b.emit();
            self.forward_beats(emitted).await?;
        } else {
            // No beats operator. Feed a pass-through into scenes by
            // synthesising `BeatsOut` variants.
            let synth: Vec<BeatsOut> = out
                .into_iter()
                .map(|s| match s {
                    SegmentOut::Segment(s) => BeatsOut::Segment(s),
                    SegmentOut::Dropped(d) => BeatsOut::Dropped(d),
                })
                .collect();
            self.forward_beats(synth).await?;
        }
        Ok(())
    }

    async fn forward_beats(&mut self, out: Vec<BeatsOut>) -> Result<()> {
        if let Some(s) = self.scenes.as_mut() {
            for item in out {
                s.ingest(item).await?;
            }
            let emitted = s.emit();
            self.absorb_scenes_out(emitted);
        } else {
            // No scenes operator; absorb manually.
            let synth: Vec<ScenesOut> = out
                .into_iter()
                .map(|b| match b {
                    BeatsOut::Segment(s) => ScenesOut::Segment(s),
                    BeatsOut::Beat(b) => ScenesOut::Beat(b),
                    BeatsOut::Dropped(d) => ScenesOut::Dropped(d),
                })
                .collect();
            self.absorb_scenes_out(synth);
        }
        Ok(())
    }

    fn absorb_scenes_out(&mut self, out: Vec<ScenesOut>) {
        for item in out {
            match item {
                ScenesOut::Segment(s) => self.drainable.segments.push(s),
                ScenesOut::Beat(b) => self.drainable.beats.push(b),
                ScenesOut::Scene(s) => self.drainable.scenes.push(s),
                ScenesOut::Dropped(d) => self.drainable.dropped.push(d),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Split a full-session track into fixed-size chunks so that one-shot
/// processing re-uses the streaming code paths. The chunk size is
/// arbitrary — it only affects intermediate allocations, not outputs,
/// per spec invariant 4.
const ONE_SHOT_CHUNK_MS: u32 = 200;

fn split_track_into_chunks(session_id: SessionId, track: SessionTrack) -> Vec<AudioChunk> {
    let samples_per_chunk = (AudioChunk::SAMPLE_RATE as u64 * ONE_SHOT_CHUNK_MS as u64 / 1000) as usize;
    let mut chunks = Vec::new();
    let mut seq = 0u32;
    let mut offset = 0usize;
    let capture_started_at: Timestamp = track.capture_started_at;
    while offset < track.pcm.len() {
        let end = (offset + samples_per_chunk).min(track.pcm.len());
        let slice: Vec<i16> = track.pcm[offset..end].to_vec();
        let duration_ms = ((slice.len() as u64) * 1000 / AudioChunk::SAMPLE_RATE as u64) as u32;
        chunks.push(AudioChunk {
            session_id,
            pseudo_id: track.pseudo_id.clone(),
            seq,
            capture_started_at,
            duration_ms,
            pcm: Arc::from(slice),
        });
        seq += 1;
        offset = end;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{OperatorKind, PipelineConfig, VadConfig};
    use crate::error::WhisperError;
    use crate::operators::vad::RmsEngine;
    use crate::types::Transcription;
    use async_trait::async_trait;
    use uuid::Uuid;

    struct MockWhisper;

    #[async_trait]
    impl WhisperClient for MockWhisper {
        async fn transcribe(
            &self,
            _audio: &[f32],
            _sr: u32,
        ) -> Result<Transcription, WhisperError> {
            Ok(Transcription {
                text: "I attack the goblin and roll for initiative".into(),
                confidence: -0.2,
                language: Some("en".into()),
            })
        }
    }

    fn tone_48k(ms: u32, freq: f32, amp: f32) -> Vec<i16> {
        let n = (48_000 * ms as usize) / 1000;
        (0..n)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                (amp * (2.0 * std::f32::consts::PI * freq * t).sin() * i16::MAX as f32) as i16
            })
            .collect()
    }

    fn silence_48k(ms: u32) -> Vec<i16> {
        vec![0; (48_000 * ms as usize) / 1000]
    }

    fn make_session() -> SessionAudio {
        let session_id = Uuid::new_v4();
        let mut pcm: Vec<i16> = Vec::new();
        pcm.extend(silence_48k(200));
        pcm.extend(tone_48k(800, 440.0, 0.5));
        pcm.extend(silence_48k(1500));
        pcm.extend(tone_48k(600, 440.0, 0.5));
        pcm.extend(silence_48k(300));
        SessionAudio {
            session_id,
            tracks: vec![SessionTrack {
                pseudo_id: "spk".into(),
                capture_started_at: 0,
                pcm: Arc::from(pcm),
            }],
        }
    }

    fn test_config() -> PipelineConfig {
        PipelineConfig {
            operators: vec![
                OperatorKind::Vad,
                OperatorKind::Transcription,
                OperatorKind::Filter,
                OperatorKind::Segment,
                OperatorKind::MetaTalk,
                OperatorKind::Beats,
                OperatorKind::Scenes,
            ],
            vad: VadConfig {
                threshold: 0.05,
                min_speech_ms: 100,
                min_silence_ms: 200,
                pad_ms: 0,
                model_path: None,
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn one_shot_produces_segments_beats_scenes() {
        let pipeline = PipelineBuilder::new(test_config())
            .whisper(Arc::new(MockWhisper))
            .vad_engine(Box::new(RmsEngine))
            .build()
            .unwrap();

        let audio = make_session();
        let result = pipeline.run_one_shot(audio).await.unwrap();

        assert!(!result.segments.is_empty(), "expected segments");
        // Each segment should carry a meta-talk tag and v7 id.
        for seg in &result.segments {
            assert!(seg.flags.meta_talk.is_some());
            assert_eq!(seg.id.get_version_num(), 7);
        }
        // Sort invariant.
        let segs = &result.segments;
        for w in segs.windows(2) {
            assert!(w[0].start_ms <= w[1].start_ms);
        }
        for w in result.beats.windows(2) {
            assert!(w[0].t_ms <= w[1].t_ms);
        }
    }

    #[tokio::test]
    async fn streaming_matches_one_shot() {
        // Canary: feeding audio in 100 ms increments must produce the
        // same segments/beats/scenes (modulo v7 UUID `id`s).
        let audio = make_session();

        // One-shot
        let pipeline = PipelineBuilder::new(test_config())
            .whisper(Arc::new(MockWhisper))
            .vad_engine(Box::new(RmsEngine))
            .build()
            .unwrap();
        let one_shot = pipeline.run_one_shot(audio.clone()).await.unwrap();

        // Streaming
        let mut streaming_pipe = PipelineBuilder::new(test_config())
            .whisper(Arc::new(MockWhisper))
            .vad_engine(Box::new(RmsEngine))
            .build()
            .unwrap();

        let session_id = audio.session_id;
        for track in audio.tracks {
            let samples_per_chunk = 48_000 / 10; // 100 ms
            let mut seq = 0u32;
            for slice in track.pcm.chunks(samples_per_chunk) {
                let pcm: Vec<i16> = slice.to_vec();
                let duration_ms = (pcm.len() as u64 * 1000 / 48_000) as u32;
                streaming_pipe
                    .ingest_chunk(AudioChunk {
                        session_id,
                        pseudo_id: track.pseudo_id.clone(),
                        seq,
                        capture_started_at: 0,
                        duration_ms,
                        pcm: Arc::from(pcm),
                    })
                    .await
                    .unwrap();
                seq += 1;
            }
        }
        let streaming = streaming_pipe.finalize().await.unwrap();

        assert_eq!(
            one_shot.segments.len(),
            streaming.segments.len(),
            "segment count diverged one_shot={:?} streaming={:?}",
            one_shot.segments.iter().map(|s| s.start_ms).collect::<Vec<_>>(),
            streaming.segments.iter().map(|s| s.start_ms).collect::<Vec<_>>(),
        );
        for (a, b) in one_shot.segments.iter().zip(streaming.segments.iter()) {
            assert_eq!(a.text, b.text);
            assert_eq!(a.pseudo_id, b.pseudo_id);
            assert_eq!(a.session_id, b.session_id);
            assert!(
                (a.start_ms as i64 - b.start_ms as i64).abs() <= 40,
                "start drift"
            );
            assert!(
                (a.end_ms as i64 - b.end_ms as i64).abs() <= 40,
                "end drift"
            );
            // UUIDs differ — that's fine.
        }
        assert_eq!(one_shot.beats.len(), streaming.beats.len());
        assert_eq!(one_shot.scenes.len(), streaming.scenes.len());
    }

    #[tokio::test]
    async fn rejects_invalid_operator_order() {
        let cfg = PipelineConfig {
            operators: vec![OperatorKind::Segment, OperatorKind::Vad],
            ..Default::default()
        };
        let err = PipelineBuilder::new(cfg)
            .whisper(Arc::new(MockWhisper))
            .vad_engine(Box::new(RmsEngine))
            .build()
            .unwrap_err();
        assert!(matches!(err, PipelineError::ConfigInvalid(_)));
    }
}
