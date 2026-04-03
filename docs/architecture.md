# OVP Pipeline — Architecture

## Purpose

A Rust library crate for audio transcription of TTRPG voice sessions. Ingests per-speaker PCM audio chunks from S3 (uploaded by the bot during recording), runs voice activity detection, transcribes via whisper.cpp, filters hallucinations, chunks into scenes, and writes transcript segments to Postgres.

Designed for reuse across the Session Helper ecosystem:
- **ttrpg-collector bot** — triggers transcription after `/stop`, or incrementally as chunks arrive (future)
- **Rust API (Axum)** — kick off transcription on demand, re-process sessions
- **Session Helper** — potential replacement for the Python streaming pipeline (future)

## Bot Audio Upload Model

The bot uploads audio to S3 **during recording**, not at `/stop`:

1. **On recording start:** bot writes `meta.json` (partial) and `consent.json` to S3 immediately
2. **During recording:** bot buffers raw PCM per speaker, uploads **50MB chunks** to S3 as they fill
3. **On `/stop`:** bot flushes final partial buffers, overwrites `meta.json` with final version (adds `ended_at`, `duration_seconds`)

### S3 Layout

```
sessions/{guild_id}/{session_id}/
  meta.json                              # Written at start, overwritten at stop
  consent.json                           # Written at start, overwritten at stop
  audio/
    {pseudo_id_a}/
      chunk_0000.pcm                     # 50MB raw PCM (s16le, 48kHz, stereo)
      chunk_0001.pcm                     # ~4.4 minutes per chunk
      chunk_0002.pcm
      chunk_0003.pcm                     # Final chunk may be smaller
    {pseudo_id_b}/
      chunk_0000.pcm
      chunk_0001.pcm
      ...
```

### Chunk Format

Raw PCM: signed 16-bit little-endian, 48kHz, 2 channels (stereo). This is exactly what Discord/Songbird delivers — zero encoding overhead during recording.

50MB = ~4.4 minutes of audio per chunk. A 3-hour session with 4 speakers produces ~160 chunks.

### Benefits

- **Crash resilience** — metadata and most audio already in S3 if bot dies mid-session
- **Pipeline can start early** — process chunks as they arrive, before session ends (future)
- **Lower peak disk usage** — chunks can be cleaned from local disk after upload
- **No ffmpeg in the bot** — no PCM→FLAC conversion step at `/stop`

## Pipeline

```
S3 chunks (per-speaker raw PCM, 50MB each)
  │
  ├─ Speaker A: chunk_0000..chunk_000N ──→ concatenate in order
  ├─ Speaker B: chunk_0000..chunk_000N ──→ concatenate in order
  ├─ Speaker C: chunk_0000..chunk_000N ──→ concatenate in order
  │                          ↓ (parallel per speaker)
  │              resample 48→16kHz (rubato)
  │                          ↓
  │                    VAD (silero via ort)
  │                          ↓
  │                    speech regions [(start, end), ...]
  │                          ↓
  └──────────────→ interleave by time across all speakers
                          ↓
                   chunk extraction (seek into PCM stream)
                          ↓
                   Whisper transcription (whisper-rs, in-process)
                          ↓
                   timestamp mapping: absolute = chunk.original_start + whisper.start
                          ↓
                   write segment to Postgres immediately
                          ↓
                   ┌─────────────────────────────┐
                   │ Filter 1: Hallucination      │ per-segment inline + periodic sweep
                   │ Filter 2: Scene Chunker      │ silence gaps + duration limits
                   └─────────────────────────────┘
                          ↓
                      Postgres (transcript_segments table)
```

## Stages

### 1. Audio Ingestion

Accepts per-speaker audio as ordered S3 chunk lists, local files, or raw bytes. No intermediate files — audio stays as sample slices in memory.

```rust
pub struct SpeakerTrack {
    pub pseudo_id: String,
    pub audio: AudioSource,
    pub sample_rate: u32,       // 48kHz from Discord
    pub channels: u16,          // 2 (stereo) from Discord
    pub bit_depth: u16,         // 16
}

pub enum AudioSource {
    /// Single file (FLAC or raw PCM)
    File(PathBuf),
    /// Single S3 object
    S3Object { bucket: String, key: String },
    /// Ordered sequence of S3 chunks — the primary ingestion path
    S3Chunked {
        bucket: String,
        prefix: String,         // "sessions/{guild}/{session}/audio/{pseudo_id}/"
    },
    /// Raw bytes in memory
    Bytes(Vec<u8>),
}
```

For `S3Chunked`, the pipeline:
1. Lists objects under the prefix
2. Sorts by chunk sequence number (lexicographic on `chunk_NNNN.pcm`)
3. Streams chunks in order, reinterpreting raw bytes as i16 samples
4. Downmixes stereo → mono (average L+R channels)

**Memory:** processes one S3 chunk at a time (~50MB = ~1.3M stereo samples). After VAD marks speech regions, only those regions are held for extraction. Peak memory is well under 100MB per speaker.

### 2. Resample

Discord audio is 48kHz. Whisper expects 16kHz. Rubato handles sample rate conversion (3:1 ratio).

Resampling happens after downmix to mono — so we're resampling one channel, not two.

No WAV, no FLAC, no intermediate files. Audio is f32 slices throughout.

### 3. Voice Activity Detection (Silero VAD)

Silero VAD is a small neural network (~2MB ONNX model) purpose-built for speech detection. Runs via `ort` (ONNX Runtime for Rust). Processes 30ms frames, outputs speech probability 0.0–1.0.

```rust
pub struct VadConfig {
    pub threshold: f32,             // Speech probability cutoff (default 0.5)
    pub min_speech_duration: f32,   // Minimum speech region seconds (default 0.25)
    pub min_silence_duration: f32,  // Silence needed to split regions (default 0.8)
    pub speech_pad: f32,            // Padding around speech (default 0.1)
}
```

**Per-speaker, parallel.** Each track runs VAD independently via `tokio::spawn`. Output is a flat `Vec<SpeechRegion>` sorted chronologically across all speakers.

```rust
pub struct SpeechRegion {
    pub start: f32,
    pub end: f32,
    pub speaker: String,
}
```

Regions shorter than `min_chunk_duration` (default 0.8s) are dropped as noise/blips.

### 4. Chunk Extraction

Given speech regions, seek into the speaker's PCM stream and extract that segment. For S3 chunked input, we know the byte offset: `offset = (start_seconds * sample_rate * channels * bytes_per_sample)`, which maps to a specific S3 chunk + position within it.

```rust
pub struct AudioChunk {
    pub speaker: String,
    pub samples: Vec<f32>,          // 16kHz f32 mono (resampled)
    pub sample_rate: u32,           // 16000
    pub original_start: f32,        // Absolute timestamp in session
    pub original_end: f32,
}
```

Chunks are ordered chronologically across all speakers.

### 5. Whisper Transcription

In-process via `whisper-rs` (bindings to whisper.cpp). Model loaded once, reused for all chunks.

```rust
pub struct TranscriberConfig {
    pub model_path: PathBuf,        // Path to ggml model file
    pub language: Option<String>,   // e.g. "en", None for auto-detect
    pub n_threads: u32,             // CPU threads for whisper.cpp (default 4)
}
```

**Timestamp mapping:**
```
absolute_start = chunk.original_start + whisper_segment.start
absolute_end   = chunk.original_start + whisper_segment.end
```

No concatenation, no drift — each chunk is independently timestamped.

**Model options:**
- `ggml-large-v3-turbo` — best quality, ~3GB, slower
- `ggml-distil-large-v3` — good balance
- `ggml-base` — fast, good for testing

Segments are written to Postgres immediately as they're produced.

### 6. Filter Chain

Two filters:

```rust
pub trait StreamFilter: Send + Sync + 'static {
    async fn on_segment(&mut self, segment: &mut TranscriptSegment) -> FilterResult;
    async fn sweep(&mut self, db: &PgPool) -> Result<u32>;
    async fn finalize(&mut self, db: &PgPool) -> Result<()>;
}

pub enum FilterResult {
    Pass,
    Exclude { reason: String },
}
```

#### Filter 1: Hallucination Detection

Frequency-based, no hardcoded phrase list.

**State:**
```rust
pub struct HallucinationFilter {
    global_counts: HashMap<String, u32>,
    per_speaker_counts: HashMap<String, HashMap<String, u32>>,
    noise_texts: HashSet<String>,
    segment_count: u32,
    pending: Vec<Uuid>,
}
```

**Inline (per segment):** empty text, repeated phrases (5+), no letters, known noise → exclude.

**Sweep (every 120s):** short text >3% frequency → noise; one speaker >80% same text → noise. Retroactively marks pending segments. Buffer drains each sweep.

#### Filter 2: Scene Chunker

```rust
pub struct SceneChunkerConfig {
    pub max_silence_gap: f32,       // Default 30.0s
    pub max_chunk_duration: f32,    // Default 600.0s
}
```

Assigns `chunk_group` to each segment for UI organization.

## Public API

### Batch Mode (process a completed session)

The happy path reads top-to-bottom. Each step takes input and returns output via `?`. No nesting, no manual error matching. If any step fails, the error propagates to the caller.

```rust
/// Process a completed recording session end-to-end.
/// Each stage is a plain function chained with `?` — the compiler
/// enforces that we don't skip steps or reorder them.
pub async fn process_session(
    config: &PipelineConfig,
    input: SessionInput,
) -> Result<PipelineResult> {
    let samples = ingest(&config.s3, &input.tracks).await?;
    let regions = detect_speech(&config.vad, &samples).await?;
    let chunks = extract_chunks(&samples, &regions, config.min_chunk_duration)?;
    let segments = transcribe(&config.whisper, &chunks).await?;
    let filtered = apply_filters(segments, &mut config.filters).await?;
    let result = commit(&config.db, input.session_id, &filtered).await?;

    Ok(result)
}
```

Each stage is a standalone async function. No wrapper types, no typestate — the function signature IS the contract. Stages are independently testable.

### Incremental Mode (process as chunks arrive — future)

```rust
pub fn create_pipeline(
    config: PipelineConfig,
) -> (PipelineSender, PipelineHandle)

impl PipelineSender {
    /// Notify that a new S3 chunk is available for a speaker
    pub async fn chunk_ready(&self, speaker: &str, chunk_key: &str) -> Result<()>;
    /// Signal that the session has ended — flush and finalize
    pub async fn finish(self) -> Result<PipelineResult>;
}
```

### Configuration

```rust
pub struct PipelineConfig {
    pub db: PgPool,
    pub s3: S3Client,
    pub whisper: TranscriberConfig,
    pub vad: VadConfig,
    pub filters: Vec<Box<dyn StreamFilter>>,
    pub min_chunk_duration: f32,    // Drop speech regions shorter than this (default 0.8s)
}

pub struct SessionInput {
    pub session_id: Uuid,
    pub tracks: Vec<SpeakerTrack>,
}

pub struct PipelineResult {
    pub segments_produced: u32,
    pub segments_excluded: u32,
    pub scenes_detected: u32,
    pub duration_processed: f32,
}
```

### Error Handling

Single error enum for the whole crate. Each stage maps its internal errors into pipeline errors via `?`. The caller gets one type to match on.

```rust
#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    #[error("ingestion failed: {0}")]
    Ingest(#[from] IngestError),
    #[error("VAD failed: {0}")]
    Vad(#[from] VadError),
    #[error("transcription failed: {0}")]
    Transcribe(#[from] TranscribeError),
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),
    #[error("S3 error: {0}")]
    S3(String),
}
```

## Crate Structure

```
ovp-pipeline/
├── Cargo.toml
├── docs/
│   └── architecture.md
├── src/
│   ├── lib.rs                  # Public API surface
│   ├── types.rs                # AudioChunk, SpeechRegion, TranscriptSegment, etc.
│   ├── pipeline.rs             # Orchestration — batch + incremental modes
│   ├── audio/
│   │   ├── mod.rs              # AudioSource, SpeakerTrack
│   │   ├── decode.rs           # Raw PCM bytes → f32 samples, stereo→mono downmix
│   │   ├── resample.rs         # 48kHz → 16kHz via rubato
│   │   ├── encode.rs           # f32 → Opus/OGG (for serving to browsers)
│   │   └── mix.rs              # Combine speaker tracks into single stream
│   ├── ingest/
│   │   ├── mod.rs              # Ingestion trait
│   │   ├── s3_chunked.rs       # List + stream ordered S3 PCM chunks
│   │   ├── file.rs             # Local file ingestion
│   │   └── memory.rs           # In-memory bytes
│   ├── vad/
│   │   └── mod.rs              # Silero VAD via ort (ONNX runtime)
│   ├── transcribe/
│   │   └── mod.rs              # whisper-rs wrapper
│   └── filters/
│       ├── mod.rs              # StreamFilter trait definition
│       ├── hallucination.rs    # Frequency-based hallucination detection
│       └── scene_chunker.rs    # Silence gap + duration-based scene boundaries
└── tests/
    ├── pipeline_test.rs
    ├── vad_test.rs
    ├── hallucination_test.rs
    └── fixtures/
        └── *.pcm              # Short test PCM clips
```

## Dependencies

| Crate | Purpose | Feature-gated |
|-------|---------|---------------|
| `rubato` | Sample rate conversion (48→16kHz) | no (core) |
| `ort` | ONNX runtime for Silero VAD | `vad` feature |
| `whisper-rs` | whisper.cpp bindings | `transcribe` feature |
| `opus` | Opus encoding for browser playback | `opus` feature |
| `ogg` | OGG container for Opus frames | `opus` feature |
| `aws-sdk-s3` | Fetch audio chunks from S3 | no (core) |
| `sqlx` | Postgres (write segments) | no (core) |
| `tokio` | Async runtime, channels, tasks | no (core) |
| `strsim` | Fuzzy string matching (future use) | no (core) |
| `uuid` | Segment/session IDs | no (core) |
| `chrono` | Timestamps | no (core) |
| `tracing` | Structured logging | no (core) |

### Feature Flags

```toml
[features]
default = ["vad", "transcribe"]
vad = ["dep:ort"]
transcribe = ["dep:whisper-rs", "dep:rubato"]
opus = ["dep:opus", "dep:ogg"]
full = ["vad", "transcribe", "opus"]
```

## Design Principles

1. **No ffmpeg.** All audio processing is native Rust. No subprocesses, no stderr parsing.

2. **No intermediate files.** Audio is f32 slices in memory. S3 chunks are streamed and processed — never written to local disk.

3. **No WAV, no FLAC in the pipeline.** Input is raw PCM (from S3 chunks). Internal representation is f32 samples. Output encoding for browsers is Opus/OGG.

4. **Happy path reads top-to-bottom.** The main `process_session` function is a chain of `?` calls. Each stage is a plain function: input → output. No nesting, no manual error matching.

5. **Stages are standalone functions.** Each stage takes typed input and returns typed output. Independently testable. No shared mutable state between stages.

6. **Filters are pluggable.** `StreamFilter` trait is the extension point. The crate ships with hallucination detection and scene chunking. Consumers add custom filters without modifying the crate.

7. **Errors propagate, don't panic.** `thiserror` enum with `#[from]` conversions. `?` everywhere. `.expect()` only in tests.

8. **Feature-gated dependencies.** Heavy deps (ONNX, whisper.cpp, libopus) behind feature flags. Default builds only pull what's needed.

9. **Iterators over loops.** Use `.filter()`, `.map()`, `.for_each()` for data transformation. Especially in the filter chain where segments flow through multiple filters.

## Bot Changes Required

The bot currently writes continuous PCM files to local disk, converts to FLAC at `/stop`, then uploads everything. The new model:

### During Recording

Replace the single `{ssrc}.pcm` file writer with a chunked buffer:

```rust
struct ChunkedWriter {
    speaker_pseudo_id: String,
    buffer: Vec<u8>,
    chunk_size: usize,          // 50MB (52_428_800 bytes)
    chunk_seq: u32,
    s3: S3Uploader,
    session_prefix: String,     // "sessions/{guild}/{session}/audio/{pseudo_id}/"
}

impl ChunkedWriter {
    fn write(&mut self, pcm_bytes: &[u8]) {
        self.buffer.extend_from_slice(pcm_bytes);
        if self.buffer.len() >= self.chunk_size {
            self.flush();       // Upload to S3, increment seq, clear buffer
        }
    }
    
    async fn flush(&mut self) {
        let key = format!("{}chunk_{:04}.pcm", self.session_prefix, self.chunk_seq);
        self.s3.upload(&key, &self.buffer).await;
        self.chunk_seq += 1;
        self.buffer.clear();
    }
}
```

### On Recording Start (quorum met)

Before any audio is captured:
1. Write `meta.json` to S3 (partial — `ended_at` and `duration_seconds` are null)
2. Write `consent.json` to S3
3. Write session + participants to Postgres (already implemented in bot DB update)

### On `/stop`

1. Flush all ChunkedWriters (upload final partial buffers)
2. Overwrite `meta.json` in S3 with final version (adds `ended_at`, `duration_seconds`, final counts)
3. Overwrite `consent.json` if any consent changed mid-session
4. Trigger pipeline: `ovp_pipeline::process_session(config, input).await`
5. Update session status in Postgres
6. Clean up local state

### No More FLAC Conversion

The bot no longer converts PCM→FLAC. Raw PCM chunks go straight to S3. The pipeline crate handles all audio processing. ffmpeg is no longer a runtime dependency of the bot.

## Consumer Examples

### ttrpg-collector bot (after /stop)

```rust
let result = ovp_pipeline::process_session(
    PipelineConfig {
        db: state.db.clone(),
        s3: state.s3_client.clone(),
        whisper: TranscriberConfig {
            model_path: config.whisper_model.clone(),
            language: Some("en".into()),
            n_threads: 4,
        },
        vad: VadConfig::default(),
        filters: ovp_pipeline::default_filters(),
        min_chunk_duration: 0.8,
    },
    SessionInput {
        session_id: bundle.session_id.parse()?,
        tracks: participants.iter()
            .filter(|p| p.consent_scope == "full")
            .map(|p| SpeakerTrack {
                pseudo_id: p.pseudo_id.clone(),
                audio: AudioSource::S3Chunked {
                    bucket: config.s3_bucket.clone(),
                    prefix: format!(
                        "sessions/{}/{}/audio/{}/",
                        guild_id, bundle.session_id, p.pseudo_id
                    ),
                },
                sample_rate: 48000,
                channels: 2,
                bit_depth: 16,
            })
            .collect(),
    },
).await?;
```

### Rust API (re-process endpoint)

```rust
// Same interface — reads chunks from S3, processes, writes segments to Postgres
ovp_pipeline::process_session(config, input).await?;
```
