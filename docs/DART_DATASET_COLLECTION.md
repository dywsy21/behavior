# DART collection interface and preflight guide

Status (2026-10-03): this is an operational interface guide for the pending
factory candidate `ece9a08203c3b7e9627d261df68e26fd36f35a33` plus
`40d79c7f008a5b3308285e297782d7f3b6307101`.  It is **not approved**, has
only CPU fixtures, and has produced no simulator trajectory or DART data.
It must not be used to accept an EULA, install Isaac/OmniGibson, download
assets, select a GPU, or run a collection job.

## Scope and gates

The future first run is one generic GRASP request.  It is candidate-only and
private: all training, actor-release, outcome, corrective-BC, recovery and
DART-success gates remain false until a separately authorized live run has
complete runtime evidence and root visual QA.

Before a command is permitted, all of the following must be real, sealed
inputs—not placeholders copied from this document:

- an independently approved factory chain, using one copy only of
  `a988 → 954 → 3c451 → 9e81 → ece9 → 40d79`;
- an authorized, compatible RTX runtime with the required NVIDIA and
  BEHAVIOR terms already accepted by the user;
- a fresh output path, source-group membership/inventory pin, runtime-session
  **plan/pins**, teacher specification/reference/receipt, calibration receipt
  and source provenance; and
- a qualified expert/teacher and a fresh noisy-state replan path; and
- an explicit run budget, stop condition, root visual-QA plan and no active
  shared-environment conflict.

The CPU-only `/home/wsy/GRM/.venv` is a test environment, not evidence that it
contains Isaac or may execute this command.

## Request contract

The JSON request has schema `p107-dart-live-grasp-request-v1`.  Its top-level
keys are exact; unknown or omitted keys fail closed:

```text
schema
pins
source
train_selection
binding
teacher_spec
teacher_reference
teacher_receipt
runtime_session
calibration
source_membership
private_writer
run_id
max_control_steps
mode
original_gaussian
bounded_dart
```

`mode` is exactly one of:

- `original_gaussian_one_pass_partial_dart`: retain the original Gaussian
  request mathematics and its calibration binding.  This is partial DART, not
  a full iterative outer learner/covariance loop.
- `dart_inspired_bounded_actual_clean_recovery`: the separate bounded,
  correlated/capped DART-inspired mode.

The unused mode block is `null`.  The request's `run_id` must equal the sealed
source collection run ID.  `private_writer` names a **new** output plus its
collector/capture/actor-observation provenance; for this live candidate its
`outcome_evidence` is explicitly `null`, so the request cannot pre-claim a
future physical outcome.  It cannot overwrite an existing publication.  Do
not create a sample request with invented paths, identifiers, hashes, source
groups, teacher receipt or calibration values.

### Reset provenance is minted, not pre-filled

The current `40d79` candidate is blocked because it accepted a
pre-filled `runtime_session.reset_load_task_instance_receipt_sha256` without
deriving it from, or comparing it with, the factory's actual `load_batch`.
Until the follow-up is independently approved, this guide is not executable.
The corrected request semantics are: `runtime_session` carries only the
expected plan/pins, rejects reset or snapshot preclaims, and its reset SHA is
`null` or absent; after its own `load_batch` and first actual RGB/state
capture, the factory must mint the reset receipt, preserve its payload and
hash in private provenance, and pass that same first capture into collection.
Any failure must publish no output.  Callers must not claim or supply a future
physical-reset receipt.

## Invocation after approval and live preflight

Use a future validated simulator interpreter and an isolated frozen checkout.
The command itself supplies no simulator default:

```bash
PYTHONPATH=src <validated-sim-python> scripts/data/collect_dart_demonstrations.py \
  --request /abs/live-grasp-request.json \
  --factory g05.recovery.dart_og_factory:collect_dart_live_candidate \
  --output /abs/fresh-candidate-summary.json
```

The CLI output is a separate candidate summary.  The factory request owns the
fresh private publication root.  A nonzero CLI exit, missing sealed input,
runtime mismatch, unsafe teacher/runtime condition, stale observation/action
receipt, or budget stop is a stop-and-report condition—not a reason to repair
pins or rerun automatically.

The pending `d995054e6e17abe8631c97c74dd524ec5217d976` follow-up additionally
requires a fresh-output preflight **before** any runtime/evaluator creation,
and idempotent cleanup of every owned post-create resource on writer/callback,
mid-run, or normal-completion paths.  This is a conditional contract pending
independent review; it does not authorize a live run.

## Private publication layout and validation

Successful factory publication is atomic at the private-writer output root:

```text
<private_writer.output>/
  publication_manifest.json
  private_collection.json
  episode/
    manifest.json
    records.jsonl
    assets/step-*/{pre,post}-{head,left_wrist,right_wrist}.png
```

`publication_manifest.json` seals `episode/manifest.json` and
`private_collection.json`.  The latter binds each captured transition's
runtime receipt, exact applied-action bytes, pre/post observation hashes and
policy/simulation clocks.  `records.jsonl` retains the distinct clean
teacher proposal, sampled noise and actual native raw23 execution provenance;
no injected noise or file existence becomes a BC-positive or physical outcome.

Validate the factory-owned publication with the approved reader, checking the
manifest/file receipts, candidate-only flags, source membership, teacher and
runtime lineage, minted reset receipt, per-transition action/clock receipts,
RGB asset hashes and freshness.  Only then may an explicitly authorized copy
for root QA be made.
No actor-facing release, training dataset, success/failed/recovery label or
DART completion claim follows from this guide or from a CPU fixture.
