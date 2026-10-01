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

## Deliberate exclusions and limits

- Renderer `a068404` is excluded pending its real float-clock fix and a new
  reviewed SHA.
- No PACKAGE synthetic test establishes frame/window uniqueness. The 10k
  scale regression reuses fixture frame/interval metadata while varying event
  identity; it tests source-episode partition accounting only.
- Candidate-only flags remain false for training, low-action supervision, and
  authority. DART clean intentions, sampled noise, and executed raw23 actions
  remain distinct.
- Any future release still needs the DATA-owned external authority, real
  temporal evidence/visual sampling, and a separately authorized live runtime.
