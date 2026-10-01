# P107 candidate-component integration receipt

Status: 2026-10-02 CST. This is a local, CPU-only integration candidate. It
creates no live trajectory, label release, data authority, simulator action,
GPU job, or training authorization.

## Frozen inputs

| Component | Accepted source | Integrated scope |
| --- | --- | --- |
| DATA contract | `dd060f3`, protocol source SHA `efdd20642fed24241f38bbdeb4abff6cf4faf1c72a86fe7c2acb32ba7496193b` | Sealed source-group membership and its 13 contract tests. |
| SIM candidate collector | `56f7452e36888f4180337c9ed7e1380cb6e7d334` | Raw candidate receipt, snapshot, evidence and direct protocol bridge, with 36 tests. |
| DART candidate collection | `99c80751ed5103e20168efdc240c6c36ce9b9603` | Original Gaussian clean-feedback and bounded actual-clean candidate paths, never a positive/release path. |
| MINING selector | `9b934f2f8dbe7ea74df184da58425921b2480fa1` | Sealed queue selector and its 9 tests. |
| PACKAGE candidate packaging | `75bf377022b1ad1308c2382e8bae9ec839a5c038` | Candidate pack/audit/dataset gate and tests; external parent authority remains separate. |
| Temporal renderer | `af1ac4d76716ed7a92c4c322ed605c98738cffa2` + `0ccbe4e46f69d75f1ee2dc05be87b4889c518eb6` | Camera-native packet renderer, including the same-frame PTS rounding fix and microsecond-future-clock rejection. |

The data compact fixture is taken from the PACKAGE revision because it carries
the current coverage expectations required by its builder. The DATA protocol
comes only from `dd060f3`; no artifact pins were rewritten.

## Integration-specific boundary

`DataSealedSourceGroupIndexReader` default-loads the local protocol by reading
and compiling its exact source bytes, not by importing `g05.data`. The latter
can initialize optional dataset dependencies that are not present in the
minimal CPU environment. The SIM bridge now translates the direct-loaded
protocol's private `ValueError`/`ContractError` into its public
`RecoveryContractError` at the typed actor-evidence boundary; this remains
fail-closed.

`tests/test_p107_integration.py` runs a `python -S` synthetic sealed-index
smoke: it confirms default DART resolution of the current DATA source SHA,
keeps `g05.data` out of `sys.modules`, verifies one load is reused after the
source-groups file is removed, and leaves the event-candidates payload absent
to demonstrate that membership does not scan events. This is provenance-only,
not a label or action pipeline.

The renderer default is the current DATA protocol source, pinned to
`efdd20642fed24241f38bbdeb4abff6cf4faf1c72a86fe7c2acb32ba7496193b`.
For a historical immutable index, `--protocol-path` requires a separate,
externally supplied SHA-256; both the input index manifest and produced packet
manifest must equal that exact reader SHA. The legacy full-v3 reader remains
the immutable snapshot
`/home/wsy/behavior-annotations/p107/protocol-snapshots/0693b93/src/g05/data/memlite_event_protocol.py`,
SHA `7f4f9fbf18fb4ba6ec97a304f0d787ebadb4f84c7184dd041c99027ad84aab0c`.
It is loaded by compiling verified source bytes, never by importing
`g05.data`; current DATA source pins were not substituted for the legacy
index's canonical-ID semantics.

## Local full-v3 metadata-only preflight

On 2026-10-02 CST, frozen integration commit
`4b4e700efa97e9b7f2673ef32306d2f3674faad5` created and resumed a fresh,
preserved external receipt directory:
`/home/wsy/behavior-annotations/p107/integration-receipts/renderer-4b4e700-20261002`.
Its renderer source SHA is
`5bcb34518b3ff77d108bb23d53e64d072e80aeb88322f25a02756c9f9d881c9c`.
This was local only (one CPU, virtual memory cap 4 GiB, 60-second command cap),
used no remote runtime and no `--decode`, and left the source index/queue
unchanged. It used the queue's required ten-position schedule
`-60,-45,-30,-15,0,1,16,31,46,60`, the legacy reader pin above, render-request SHA
`b5a76f786ad38adbc17b25649899505d25951c7d4efa437f957856c08b757edb`,
and queue-seal SHA
`78eda9c87b18e806e00e4820c172b02002d806e7c77fe9911368d42711de227c`.

Creation completed in 7.72 seconds (28,252 KiB max RSS) with 40 candidate
packets, 400 requested temporal slots, 392 distinct source frames, and 8
boundary-clamped duplicate requests; it wrote zero rendered RGB receipts and
remained `LOCATORS_READY_RENDER_PENDING`. The new external resume pin is
`fa4d23d798564babbbf2fd87d6aae6150cf53b6d50f3066f57f830e158cbbd47`;
resume completed in 6.28 seconds (28,752 KiB max RSS) as
`RESUME_VALIDATED`. These are renderer/provenance receipts, not an image
review, action label, data release, or training result. The receipt directory
preserves the exact command, script/input SHA list, stdout, runtime files, and
sealed packet manifest/payloads for independent revalidation.

## Deliberate exclusions and limits

- No PACKAGE synthetic test establishes frame/window uniqueness. The 10k
  scale regression reuses fixture frame/interval metadata while varying event
  identity; it tests source-episode partition accounting only.
- Candidate-only flags remain false for training, low-action supervision, and
  authority. DART clean intentions, sampled noise, and executed raw23 actions
  remain distinct.
- Any future release still needs the DATA-owned external authority, real
  temporal evidence/visual sampling, and a separately authorized live runtime.
