# P107 DART candidate collection design

Status: 2026-10-01 Beijing time, implementation candidate only.  This design
creates no data, opens no simulator, uses no GPU, and mints no training or
release authority.  Live collection remains blocked on an officially
compatible RTX simulator route, the user's 4080 decision, and repaired VPN
access; lc1/lc2 must not be disturbed.

## What DART means here

The original algorithm is **not** offline jitter.  Laskey et al. define a
noisy supervisor action distribution `a ~ N(pi*(x), Sigma)` in Eq. 2, optimize
the likelihood of clean supervisor actions at states visited under that
distribution (Eq. 3), and scale the covariance by
`alpha/(T trace(Sigma))` (Eq. 4).  See [the original paper, pp. 3--4](https://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf) and the authors'
[Gaussian supervisor](https://github.com/BerkeleyAutomation/DART/blob/master/experiments/tools/supervisor.py),
[trajectory collection](https://github.com/BerkeleyAutomation/DART/blob/master/experiments/tools/statistics.py),
and [per-trajectory covariance estimator](https://github.com/BerkeleyAutomation/DART/blob/master/experiments/tools/noise.py).
The released code stores both intended and taken actions, executes the sampled
action, and uses the intended supervisor action as the learner label.

This branch implements two deliberately non-interchangeable candidate views:

| View | `label_kind` | Execution and label meaning | Training status |
| --- | --- | --- | --- |
| Original-Gaussian clean feedback | `dart_clean_supervisor_feedback` | At every current state reached under prior noisy controls, re-query the teacher; record its one clean intended 23-D target and execute one unbounded Gaussian sample.  Exact mode rejects `applied23 != requested_noisy23`. | Candidate only.  The clean target was not asserted to be actual execution. |
| Safety-bounded recovery | `dart_inspired_noisy_injection` then `dart_inspired_actual_clean_recovery` | Named native parts receive bounded, causal temporally correlated noise; every later recovery control is re-queried and must be applied exactly as the clean teacher action. | Both views candidate only; noisy injection is explicitly excluded from FM supervision. |

`frozen_one_pass_partial_dart` records Eq. 2 sampling with a frozen covariance,
but marks `empirical_faithfulness=false`; it cannot substantiate the original
iterative DART empirical claim.  The present candidate collector rejects a
caller request for `full_iterative_covariance_update` and has no code path that
can set empirical faithfulness true.  A separately implemented, fully logged
iterative collection/covariance update loop would require fresh review before
it could make that claim.  Clipping, named-part selection, caps and temporal
correlation are explicitly
`dart_inspired_bounded_correlated`, never “original DART.”

## Receipt and safety contract

Each candidate record has all of the following, with immutable SHA receipts:

- `clean_intended23`, `requested_noisy23`, `sampled_noise23`, and `applied23`
  are separate fields.  `applied_action_bytes_sha256` comes from the backend,
  rather than reconstructing a claimed physical execution from Python floats.
  The applied receipt repeats the exact pre-action clock/state/observation
  digests and is rejected unless they match the fresh teacher-query state.
- Exact current `policy_clock`, fresh RGB/proprio state fingerprint,
  observation fingerprint/ref, sealed TRAIN parent task/task-instance/seed,
  fresh DART trajectory source kind plus collection-run/reset identity, frozen
  teacher code/weights/config/runtime, noise profile and sampling seed are
  recorded.  The new reached state is never assigned an original demo episode
  identity merely because its parent task-instance is in the sealed index.
  `TeacherReceipt` requires `feedback_mode=closed_loop_fresh_observation` and
  every target has a fresh-query receipt.  An original-demo action array cannot
  masquerade as a new-state expert response.  A teacher response whose clock,
  state or observation receipt differs from the present reached state is
  rejected as a future leak; `label_source` explicitly identifies human,
  planner, frozen-policy, or privileged-oracle labels.  Privileged oracle state
  may label a candidate but never becomes actor input.  The teacher receipt
  also fingerprints its actual postcondition specification; a candidate without
  later runtime postcondition evidence cannot become an eventual release.
- A clean label is one step only.  The record states the model contract
  (predict 32, execute at most 16 from start 0) but never packs recommendations
  from different reached states into a fake 32-step clean expert future.
- Candidate source groups and reserved calibration groups are verified by the
  DATA-owned, externally SHA-pinned source index API
  (`DataSealedSourceGroupIndexReader` over protocol commit `dd060f3`), not by
  caller strings.  It loads and seals the membership index once, then requires
  a canonical `student_candidate` TRAIN group and distinct canonical
  `annotation_calibration` TRAIN groups.  The adapter must name the exact 116
  held-out **TRAIN** calibration groups from the full-v3 index in
  `CalibrationReceipt`; public/dev/eval and protected holdouts are rejected
  rather than re-used.  This membership reader provides no training, positive
  label, or release authority.
- Original-Gaussian collection additionally binds the exact sampled covariance
  to a calibration trajectory SHA, learner/teacher checkpoint SHAs, covariance
  estimator-code SHA, `alpha`, horizon, and sealed source-index manifest SHA.
  A covariance that does not equal the Eq. 4 scaling of that artifact is
  rejected before collection.
- The frozen-v4 metadata maps immutable TRAIN task/task-instance source groups,
  but has no simulator snapshot/restore or controller/grasp/particle/RNG state.
  `RuntimeSessionReceipt` therefore requires a fresh official
  reset→load-task-instance receipt plus runtime and asset/config hashes that
  match the source task instance.  Any later same-state branch must identify a
  newly captured, verified snapshot in that exact live session; demo video or
  action replay cannot stand in for it.
- Native noise receives a metadata-derived, named 23-D physical layout with
  immutable `embodiment_metadata_sha256` and
  `model_projection_manifest_sha256` receipts.  There are no raw-index
  constants and no 27-D padded controls.  Every physical part, including base
  and trunk, needs an explicit perturb or non-perturb reason; non-perturbation
  does not disable its applied controller action.  The known model padding
  `[7,8,17,18]` stays inactive only after projection.
- An abort, nonfinite action, invalid PSD covariance, mismatched actual action
  in exact mode, or an over-16 recovery chunk aborts collection.  Delayed
  outcome receipts are only `ACTUAL_FAULT`, `SURVIVAL_NONFAILURE`, or
  `OUTCOME_UNKNOWN`, carry observed/available clocks, and never enter actor
  input.  A cap/budget stop remains `OUTCOME_UNKNOWN`.

All records keep `candidate_only=true`, `training_eligible=false`,
`ready_for_training=false`, `low_action_supervision_positive=false`, and
`authority_minted=false`, with `authority_status=NO_AUTHORITY`.  The CLI
checks each of those exact values on the collection result and every record,
even when a record hash is otherwise valid.  The existing `corrective_action` view remains
actual-execution-only and is not changed by this work.

## Interfaces and command

`src/g05/recovery/dart_collection.py` is an additive closed-loop seam, with no
simulator imports:

```text
DartRuntime.observe() -> DartObservation
DartTeacher.clean_action(observation, intent_bundle_id) -> TeacherCommand
DartRuntime.apply_raw23(requested23) -> AppliedActionReceipt
```

An adapter must supply a frozen teacher receipt.  It may be a human or planner;
a stage-1 checkpoint is provisional, not an oracle.  The future simulator
owner can wrap its owned runtime after independent review; this branch does not
modify `PairedRecoveryCollector` or launch hardware/software jobs.

```bash
PYTHONPATH=src python scripts/data/collect_dart_demonstrations.py \
  --request candidate-request.json \
  --factory integration_module:build_candidate_result \
  --output candidate-result.json
```

The factory is mandatory.  The CLI only writes a new (`x` mode) candidate JSON
after hash and no-authority checks; it never chooses a runtime, installs an
environment, connects SSH, or starts a job.

## Bounded engineering pilot registration (not launched)

| Field | Registration |
| --- | --- |
| Hypothesis | The adapter can faithfully maintain the clean-label/noisy-execution/actual-clean distinction and causal one-step re-planning; this does **not** claim a learning or recovery-success effect. |
| Code/data | This branch at its final committed SHA; full-v3 source manifest plus a frozen exact candidate source-group list.  Calibration is the declared 116 held-out TRAIN groups, never protected eval/dev. |
| Teacher | Exact human/planner identity, code/weights/config/runtime SHA in `TeacherReceipt`; no presumed oracle or final stage-1 checkpoint. |
| Tasks/seeds/reset | Adapter must declare sealed parent task/task-instance/seed, the fresh DART collection run, intent, and exact reset identity/receipt before execution.  No reset has been performed. |
| Control/budget | Compare matched no-noise expert and DART-noise+expert candidates from the same TRAIN source-group families.  Replan causally each control; chunks are 1--16 steps, with declared total-step and wall-time caps. |
| Stop conditions | Abort/missing receipt/nonfinite/teacher mismatch/physical safety stop/route incompatibility; budget expiry is unknown outcome.  Do not infer semantic recovery from an engineering loop. |
| Compute/live state | 0 GPU, 0 SSH, 0 installation, 0 simulator or robot actions.  Official RTX compatibility, 4080 user permission and VPN repair remain dependencies. |

Independent review must inspect the source-group and calibration manifests,
teacher/runtime receipts, candidate hashes, noisy vs actual audit trail, and
raw video/state evidence before any external data authority considers a
release.  The prior candidate commit `569200e` was independently marked
NEEDS-FIX; this follow-up addresses its flag, covariance-faithfulness,
membership/calibration-binding, and layout-provenance findings but remains
candidate-only and pending another independent review.  Parent-agent
stratified visual sampling remains required.  Integration must carry only the
new DART-owned files onto SIM `56f7452` (or its reviewed successor), rather
than reviving the obsolete bridge or stale planning documents from the earlier
candidate branch.
