# ovp-pipeline

Rust library crate for TTRPG voice session transcription. Ingests per-speaker PCM audio, runs VAD, transcribes via whisper.cpp, filters hallucinations, and writes segments to Postgres.

## Documentation

| Document | What |
|----------|------|
| `docs/architecture.md` | Pipeline stages, data flow, crate structure, dependencies |

## Code Style

Follow [Rust Design Patterns](https://rust-unofficial.github.io/patterns/) and [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).

**Readability:**
- Add comments that explain **why**, not what. Every public function should have a doc comment explaining its purpose and rationale.
- Pipeline stages and filter logic should have inline comments — especially audio processing decisions, timing thresholds, and memory management choices.
- Keep comments concise but sufficient for someone reading the code for the first time.

**Patterns:**
- Use `Result` + `?` for error propagation — the happy path reads top-to-bottom like pseudocode.
- Pipeline stages are plain functions chained with `?`. No unnecessary wrapper types.
- Filters implement the `StreamFilter` trait — pluggable, composable.
- Use iterators over manual loops. Use `filter`, `map`, `for_each`.
- Avoid premature abstraction — simple functions over complex generics.

**Anti-patterns:**
- No nested if/match chains — flatten with `?` and early returns.
- No scattered state — each pipeline stage takes input and returns output.
- No manual mutex on hot paths — use channels for concurrent data flow.
