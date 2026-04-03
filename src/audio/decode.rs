//! Raw PCM decoding and stereo-to-mono downmix.
//!
//! Takes raw s16le PCM bytes from a `SpeakerTrack` and produces
//! mono f32 samples suitable for downstream processing.

use crate::error::{PipelineError, Result};
use crate::types::{SpeakerSamples, SpeakerTrack};

/// Decode raw s16le PCM bytes into mono f32 samples.
///
/// Each pair of bytes is interpreted as a signed 16-bit little-endian
/// sample. For stereo tracks, left and right channels are averaged
/// to produce mono output.
pub fn decode(tracks: &[SpeakerTrack]) -> Result<Vec<SpeakerSamples>> {
    tracks.iter().map(decode_track).collect()
}

/// Decode a single speaker track from raw PCM to mono f32 samples.
fn decode_track(track: &SpeakerTrack) -> Result<SpeakerSamples> {
    // Each sample is 2 bytes (s16le)
    if track.pcm_data.len() % 2 != 0 {
        return Err(PipelineError::Decode(format!(
            "speaker {}: PCM data length {} is not a multiple of 2 bytes",
            track.pseudo_id,
            track.pcm_data.len()
        )));
    }

    let channels = track.channels as usize;
    if channels == 0 {
        return Err(PipelineError::Decode(format!(
            "speaker {}: channel count cannot be zero",
            track.pseudo_id
        )));
    }

    // Parse all s16le samples into f32 in [-1.0, 1.0]
    let all_samples: Vec<f32> = track
        .pcm_data
        .chunks_exact(2)
        .map(|pair| {
            let sample = i16::from_le_bytes([pair[0], pair[1]]);
            sample as f32 / i16::MAX as f32
        })
        .collect();

    // Downmix to mono by averaging across channels per frame
    let total_samples = all_samples.len();
    if total_samples % channels != 0 {
        return Err(PipelineError::Decode(format!(
            "speaker {}: sample count {} is not a multiple of channel count {}",
            track.pseudo_id, total_samples, channels
        )));
    }

    let mono_samples: Vec<f32> = if channels == 1 {
        all_samples
    } else {
        all_samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    tracing::debug!(
        speaker = %track.pseudo_id,
        input_bytes = track.pcm_data.len(),
        mono_samples = mono_samples.len(),
        sample_rate = track.sample_rate,
        "decoded PCM track"
    );

    Ok(SpeakerSamples {
        pseudo_id: track.pseudo_id.clone(),
        samples: mono_samples,
        sample_rate: track.sample_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_mono_silence() {
        let track = SpeakerTrack {
            pseudo_id: "test".into(),
            pcm_data: vec![0, 0, 0, 0, 0, 0, 0, 0],
            sample_rate: 48000,
            channels: 1,
        };
        let result = decode(&[track]).unwrap();
        assert_eq!(result[0].samples.len(), 4);
        assert!(result[0].samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn decode_stereo_downmix() {
        // Two frames of stereo: L=max, R=0 -> mono = 0.5
        let max_bytes = i16::MAX.to_le_bytes();
        let zero_bytes = 0i16.to_le_bytes();
        let mut pcm = Vec::new();
        pcm.extend_from_slice(&max_bytes);
        pcm.extend_from_slice(&zero_bytes);
        pcm.extend_from_slice(&max_bytes);
        pcm.extend_from_slice(&zero_bytes);

        let track = SpeakerTrack {
            pseudo_id: "test".into(),
            pcm_data: pcm,
            sample_rate: 48000,
            channels: 2,
        };
        let result = decode(&[track]).unwrap();
        assert_eq!(result[0].samples.len(), 2);
        assert!((result[0].samples[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn decode_odd_bytes_fails() {
        let track = SpeakerTrack {
            pseudo_id: "test".into(),
            pcm_data: vec![0, 0, 0],
            sample_rate: 48000,
            channels: 1,
        };
        assert!(decode(&[track]).is_err());
    }
}
