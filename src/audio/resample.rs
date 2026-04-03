//! Sample rate conversion via rubato.
//!
//! Resamples mono f32 audio from one sample rate to another (typically
//! 48kHz -> 16kHz for Whisper). Operates on already-downmixed mono samples.

use crate::error::{PipelineError, Result};
use crate::types::SpeakerSamples;

/// Resample all speaker samples from `from_rate` to `to_rate`.
///
/// Requires the `transcribe` feature (which pulls in rubato).
/// Without it, returns an error.
pub fn resample(
    samples: &[SpeakerSamples],
    from_rate: u32,
    to_rate: u32,
) -> Result<Vec<SpeakerSamples>> {
    // No-op if rates already match
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    samples
        .iter()
        .map(|s| resample_speaker(s, from_rate, to_rate))
        .collect()
}

#[cfg(feature = "transcribe")]
fn resample_speaker(
    speaker: &SpeakerSamples,
    from_rate: u32,
    to_rate: u32,
) -> Result<SpeakerSamples> {
    use rubato::{FftFixedIn, Resampler};

    if speaker.sample_rate != from_rate {
        return Err(PipelineError::Resample(format!(
            "speaker {}: expected sample rate {} but got {}",
            speaker.pseudo_id, from_rate, speaker.sample_rate
        )));
    }

    if speaker.samples.is_empty() {
        return Ok(SpeakerSamples {
            pseudo_id: speaker.pseudo_id.clone(),
            samples: Vec::new(),
            sample_rate: to_rate,
        });
    }

    // rubato chunk size — 1024 is a reasonable default
    let chunk_size = 1024;
    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
        1, // sub-chunks
        1, // mono channel
    )
    .map_err(|e| PipelineError::Resample(format!("speaker {}: {}", speaker.pseudo_id, e)))?;

    let mut output = Vec::with_capacity(
        (speaker.samples.len() as f64 * to_rate as f64 / from_rate as f64) as usize + chunk_size,
    );

    // Process full chunks
    let input = &speaker.samples;
    let mut pos = 0;

    while pos + chunk_size <= input.len() {
        let chunk = vec![input[pos..pos + chunk_size].to_vec()];
        let resampled = resampler
            .process(&chunk, None)
            .map_err(|e| PipelineError::Resample(format!("speaker {}: {}", speaker.pseudo_id, e)))?;
        output.extend_from_slice(&resampled[0]);
        pos += chunk_size;
    }

    // Handle remaining samples by zero-padding to chunk_size
    if pos < input.len() {
        let mut last_chunk = input[pos..].to_vec();
        last_chunk.resize(chunk_size, 0.0);
        let chunk = vec![last_chunk];
        let resampled = resampler
            .process(&chunk, None)
            .map_err(|e| PipelineError::Resample(format!("speaker {}: {}", speaker.pseudo_id, e)))?;

        // Only take the proportional amount of resampled output
        let expected_remaining =
            ((input.len() - pos) as f64 * to_rate as f64 / from_rate as f64).ceil() as usize;
        let take = expected_remaining.min(resampled[0].len());
        output.extend_from_slice(&resampled[0][..take]);
    }

    tracing::debug!(
        speaker = %speaker.pseudo_id,
        input_samples = input.len(),
        output_samples = output.len(),
        from_rate,
        to_rate,
        "resampled audio"
    );

    Ok(SpeakerSamples {
        pseudo_id: speaker.pseudo_id.clone(),
        samples: output,
        sample_rate: to_rate,
    })
}

#[cfg(not(feature = "transcribe"))]
fn resample_speaker(
    speaker: &SpeakerSamples,
    _from_rate: u32,
    _to_rate: u32,
) -> Result<SpeakerSamples> {
    Err(PipelineError::Resample(format!(
        "speaker {}: resampling requires the `transcribe` feature (rubato)",
        speaker.pseudo_id
    )))
}
