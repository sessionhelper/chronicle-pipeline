# chronicle-pipeline

> Org-wide conventions (Rust style, git workflow, cross-service architecture) live in `/home/alex/sessionhelper-hub/CLAUDE.md`. Read that first for anything cross-cutting.

Pure Rust library. PCM `f32` in → `TranscriptSegment`s out. No I/O except the outbound Whisper HTTP call. No Postgres, no S3, no filesystem.

## Pipeline stages

1. **Resample** to 16 kHz (Rubato).
2. **RMS gate** — cheap tier-1 silence filter.
3. **Silero VAD v6** via ONNX (`ort`) — tier-2 speech detection. Model file `models/silero_vad_v6.onnx` is gitignored; must exist at runtime.
4. **Whisper** transcription via HTTP endpoint (e.g. faster-whisper, OpenAI-compatible).
5. **Operator chain** — pluggable post-processors:
   - `HallucinationFilter` — frequency-based, cross-speaker dedup
   - `SceneChunker` — silence-gap-based grouping
   - `SceneOperator` — optional LLM-backed boundaries
   - `BeatOperator` — narrative beat grouping

## Public API

```rust
pub async fn process_session(
    config: &PipelineConfig,
    input: SessionInput,
    operators: &mut [Box<dyn Operator>],
) -> Result<PipelineResult>;
```

## Repo-specific conventions

- **Library, not a binary.** Keep I/O out except the Whisper HTTP call.
- Feature-gated heavy deps: `vad` (ONNX), `transcribe` (reqwest), `cli` (test scaffold binary).
- Operators implement the `Operator` trait — single-method, composable, no shared state.
- See `docs/architecture.md` for full pipeline diagram.

## Build

```bash
cargo check
cargo test
cargo build --release --features "vad,transcribe"
```
