# P107 phase-balanced calibration candidates (metadata only)

This note records a bounded, candidate-only follow-up to the v1 calibration
queue. It is not an outcome dataset, recovery dataset, action dataset, or RGB
rendering approval.

## Immutable input and output

- Parent index: `/home/wsy/behavior-annotations/p107/index-validation/full-v3-candidate-index`
  - inventory seal: `7ed1b2c5cbb50c8af042a2b9dca8529a6d133de5fc0fbe224601854235c31479`
  - frozen source manifest: `90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23`
  - owner protocol: `7f4f9fbf18fb4ba6ec97a304f0d787ebadb4f84c7184dd041c99027ad84aab0c`
- New metadata-only output: `/home/wsy/behavior-annotations/p107/phase-balanced-calibration40-v2`
  - phase queue seal: `400613905529c710fcba62ebe4b7167c6df974ada94256d05a0c7bdc94b46c8f`
  - selection manifest: `ee02b54d617100c6d2f6526e4527b546285a324051b79a41872fd54dc5397b5c`
  - renderer-compatible mini-index manifest: `a43bbec04a86d0b13f9ad06f4825fcd7c16ab469cf13c3a65a555a5ab0eca09b`

The producer streams the sealed parent index, retains only the immutable
`train`/`annotation_calibration` role, and has no RGB, simulator, GPU, or
training path. The completed production invocation was pinned to one CPU and
took 17.3 seconds; sealed resume reconstruction took 17.0 seconds.

## Selection contract and completed counts

Exactly 40 new event IDs were projected from the original source skill and
segment bounds, with one selected source group and source episode per row:

| Selection stratum | Observation phase | Rows |
| --- | --- | ---: |
| `ENTRY` | `ENTRY` | 10 |
| `MID` | `MID` | 10 |
| `TERMINAL_OR_TRANSITION` | `TERMINAL_TRANSITION` | 10 |
| `REPEATED_METADATA_QUERY` | `REPEATED_METADATA_QUERY` | 10 |

All 40 rows are immutable `train` / `annotation_calibration` source records
with `MISSING` evidence. They cover 31 tasks and 31 official skills. The
selection manifest records missing selection coverage: task IDs other than the
31 chosen rows and skill IDs `94`, `100`, `101`, and `103`.

Every derived row records its original queried-skill start/end, parent event
ID, parent member index, and raw member digest. Terminal/transition anchors
are at `segment_end - 1` and query the original member whose own end is that
segment end; the following parent event is lineage only. An intent-start frame
is exposed as metadata for a possible later causal renderer, but this queue
does not say that any pre-anchor images/history have been rendered.

`REPEATED_METADATA_QUERY` is only a same task/skill metadata key occurring in
at least two distinct source episodes (the selected rows have 3--5 such
episodes). It is not an attempt, retry, recovery, success, or failure label.
Observation phase is likewise a source-clock coordinate, not an outcome.

## Compatibility and gates

The mini-index uses `memlite-event-index-v1`, has canonical new phase-event
IDs, and was preflighted directly through the existing renderer's legacy
protocol binding and camera-locator checks. No adapter, packet directory, or
RGB decode was created. The repository regression test also creates a sealed
40-row mini-index, validates resume, and exercises a locator-only direct
renderer read.

Before any decode, an independent reviewer must inspect the sealed selection
manifest, the phase queue, and candidate mini-index. That review must keep the
output candidate-only and must not infer success, failure, or recovery from a
segment boundary or repeated metadata.
