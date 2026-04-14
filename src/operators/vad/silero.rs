//! Silero VAD v6 ONNX engine.
//!
//! Implements [`VadEngine`] against the bundled Silero ONNX model at
//! 16 kHz. Caller provides the path via [`VadConfig::model_path`].

use std::path::PathBuf;

use ort::session::Session;
use ort::value::Tensor;

use super::{VadContext, VadEngine, FRAME_SIZE};
use crate::error::PipelineError;

const LSTM_DIM: usize = 128;
const BATCH: usize = 16;

pub struct SileroEngine {
    model_path: PathBuf,
}

impl SileroEngine {
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path }
    }
}

impl VadEngine for SileroEngine {
    fn new_context(&self) -> Box<dyn VadContext> {
        Box::new(SileroContext::open(self.model_path.clone()))
    }
}

pub struct SileroContext {
    session: Option<Session>,
    #[allow(dead_code)]
    model_path: PathBuf,
    h: Vec<f32>,
    c: Vec<f32>,
    open_error: Option<String>,
}

impl SileroContext {
    fn open(model_path: PathBuf) -> Self {
        let result = Session::builder()
            .and_then(|mut b| b.commit_from_file(&model_path));
        match result {
            Ok(session) => Self {
                session: Some(session),
                model_path,
                h: vec![0.0; LSTM_DIM],
                c: vec![0.0; LSTM_DIM],
                open_error: None,
            },
            Err(e) => {
                let msg = format!("load {:?}: {e}", &model_path);
                Self {
                    session: None,
                    model_path,
                    h: vec![0.0; LSTM_DIM],
                    c: vec![0.0; LSTM_DIM],
                    open_error: Some(msg),
                }
            }
        }
    }
}

impl VadContext for SileroContext {
    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, PipelineError> {
        if let Some(err) = &self.open_error {
            return Err(PipelineError::Vad(err.clone()));
        }
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| PipelineError::Vad("silero session not initialized".into()))?;

        let total_frames = samples.len() / FRAME_SIZE;
        if total_frames == 0 {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(total_frames);
        let mut done = 0;
        while done < total_frames {
            let batch_len = BATCH.min(total_frames - done);

            let mut input = vec![0.0f32; batch_len * FRAME_SIZE];
            for i in 0..batch_len {
                let src = (done + i) * FRAME_SIZE;
                input[i * FRAME_SIZE..(i + 1) * FRAME_SIZE]
                    .copy_from_slice(&samples[src..src + FRAME_SIZE]);
            }

            let input_t =
                Tensor::from_array(([batch_len, FRAME_SIZE], input.into_boxed_slice()))
                    .map_err(|e| PipelineError::Vad(format!("input tensor: {e}")))?;
            let h_t = Tensor::from_array(([1usize, 1, LSTM_DIM], self.h.clone().into_boxed_slice()))
                .map_err(|e| PipelineError::Vad(format!("h tensor: {e}")))?;
            let c_t = Tensor::from_array(([1usize, 1, LSTM_DIM], self.c.clone().into_boxed_slice()))
                .map_err(|e| PipelineError::Vad(format!("c tensor: {e}")))?;

            let outputs = session
                .run(ort::inputs!["input" => input_t, "h" => h_t, "c" => c_t])
                .map_err(|e| PipelineError::Vad(format!("silero infer: {e}")))?;

            let (_, probs) = outputs["speech_probs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| PipelineError::Vad(format!("extract probs: {e}")))?;
            out.extend(probs.iter().copied());

            let (_, hn) = outputs["hn"]
                .try_extract_tensor::<f32>()
                .map_err(|e| PipelineError::Vad(format!("extract hn: {e}")))?;
            let (_, cn) = outputs["cn"]
                .try_extract_tensor::<f32>()
                .map_err(|e| PipelineError::Vad(format!("extract cn: {e}")))?;
            self.h.copy_from_slice(&hn[..LSTM_DIM]);
            self.c.copy_from_slice(&cn[..LSTM_DIM]);

            done += batch_len;
        }
        Ok(out)
    }
}
