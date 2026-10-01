# P107: privileged-pose GRASP DART adapter seam

Status: CPU contract only; candidate-only.  No OmniGibson import, license
acceptance, simulator launch, GPU work, robot action, DART record, data release
or training authorization occurred.

## Narrow hypothesis

For a fresh official TRAIN rollout, one generic GRASP controller can re-query a
privileged, current-state pose planner after every noisy reached state and emit
one native R1Pro raw23 clean target.  This is an engineering hypothesis, not a
claim that an existing model is an oracle, that recovery succeeded, or that a
candidate is trainable.

`src/g05/recovery/privileged_pose_grasp.py` is deliberately a seam.  The future
LC owner must inject the already reviewed `PoseTeacher`, `SafeServo`,
`PrivilegedReader`, and `LocalOutcome` implementations with their code/config
hashes into `TeacherReceipt`.  It supplies no simulator factory and does not
turn a fake hook into a live adapter.

## Data and authority boundary

`SealedGraspBinding` accepts only canonical `student_candidate` or disjoint
`annotation_calibration` groups with `original_split=train`.  It binds one
causal intent, target, hand, source-member/index-manifest hashes and static
object-local pose-seed hash; it can cross-check the DATA canonical membership
object already verified by the DART collector.  The latter
must carry `RootModelPoseReview`, which hard-codes
`human_reviewed=false` and `reviewer_kind=root_model_visual_geometry`; it is
not a manual teaching approval and cannot be emitted as a human label.
`GenericGraspPilot` rejects a one-object cohort, duplicate source groups, or a
single target category, while pinning one controller code/config hash across
the whole cohort; this prevents a collection of per-object task scripts from
being described as a generic controller.

Actor projection accepts only pre-rendered task/causal text whose bytes and
description-registry hash match the sealed binding, plus a hash-only current
RGB/proprio media reference.  It does not forward `DartObservation.public()`:
free-form observation paths and raw target/source identifiers are rejected or
omitted.  Exact object identity, world/object/hand poses, contacts, grasp
state, goal pose and all postconditions remain teacher-only.  Each clean
command contains the current policy clock/state/observation hash, an opaque
fresh query hash, and a one-step raw23 target.  Demonstration actions and
future frames do not enter this path.

## Control and physical evidence

`ExistingPoseServoEngine` holds one native `PoseTeacher` per sealed binding
session.  It ranks current pose actions and asks `SafeServo` only to preview
the first finite native command; a preview never advances its native lifecycle.
Only an exact actual raw23 receipt from `FreshRolloutRaw23Runtime` may
acknowledge that proposal and call native `PoseTeacher.executed`.  A different
noisy action normally discards the preview.  The one narrow exception is a
native `LEFT_CLOSE`/`RIGHT_CLOSE` whose actual action preserves **both** R1Pro
gripper raw23 float32 wire channels (14 and 22) byte-for-byte: it may advance
the private native lifecycle after the same fresh receipt checks, but remains a
noisy actual execution—not a clean BC target, trusted action, or evidence
trace.  Any changed gripper channel or failed/unknown noisy CLOSE lifecycle
terminally retires the episode; a subsequent fresh `held=FALSE` cannot request
another CLOSE.
The engine cannot switch bindings mid-session or infer a previous CLOSE from a
fresh `held` flag.  `FreshRolloutRaw23Runtime` is the sole execution seam; it
accepts no snapshot mode other than `no_restore_fresh_rollout`, requires a
fresh preceding observation, exact applied raw23 equality and the canonical
little-endian float32 action-byte SHA.

The local GRASP condition is pinned in `GraspPostconditionConfig` and passed to
the existing `LocalOutcome` verifier only through a private trace opened from
a runtime-acknowledged clean action.  Bare caller-provided frames and
`RIGHT_CLOSE` strings are candidate-only `UNKNOWN`, not local certification.
The trace binds the selected native token, exact raw23 bytes and the clean
command's pre-action clock/state/observation hashes to the consumed runtime
receipt.  It requires current target identity,
initial empty hands, correct-hand TRUE grasp plus target contact, target lift
at least 3 cm, hand lift at least 2.5 cm, no forbidden contact/payload loss,
and stable target-in-hand relative pose (4 mm / 3 degrees).  Stability requires
at least 12 distinct contiguous physics ticks **and** at least 0.5 seconds,
with a pinned timestep and a required runtime physics-clock receipt on every
frame; timestamps must be strictly increasing at exactly that timestep.
`physics_dt_seconds` and its receipt must be taken from the bound runtime, not
a unit-fixture default.  H75 historically advanced roughly four physics ticks
per control call, so a future hook that reads only once per control is not a
contiguous-tick recorder: it must collect per-physics-tick evidence (or remain
`UNKNOWN`) rather than silently treating control samples as physics samples.
Missing contact/grasp/timestamp/clock evidence is `UNKNOWN`, never inferred
solid contact.
The resulting receipt explicitly has `official_task_success=false` and is not
an action-BC or outcome release.

## Unfinished live hook

An LC-owned implementation still has to bind a freshly reset official task
instance to `FreshRuntimeHooks.observe_current` and
`apply_current_raw23`, and bind `PrivilegedWorld.fresh_snapshot` to the same
session's `PrivilegedReader` / kinematics / calibrated servo model.  It must
then demonstrate two or more target object categories across TRAIN groups,
fresh query-after-noise, raw action-byte receipts, and the local physical
postcondition.  It also needs a private per-physics-tick recorder that stamps
the runtime clock receipt; `FreshRuntimeHooks` currently has no such live
method, so this remains an unfinished LC hook rather than a claim of live
certification.  No snapshot restore is required or enabled for this first
pilot.  Any missing official API, action substitution, stale hash, no eligible
candidate, unknown required grasp/contact/clock sensor or unsafe physical
contact must abort/mask the candidate rather than fabricate an expert label. Ordinary
`held=FALSE` during approach is expected and remains re-plannable; bounded
attempts retain `IN_PROGRESS`, `FAILED` or `UNKNOWN` local diagnostics without
calling any of them a clean teacher label or inferring failure from a timeout or
annotation boundary.

## CPU validation

`PYTHONPATH=src python tests/test_privileged_pose_grasp.py` covers R1Pro mapping
and padding rejection, current-state re-query, preview-only teacher behavior,
native CLOSE acknowledgement versus unapplied proposal, stale/exact runtime
receipts, no-human provenance, TRAIN-only role rejection, binding-aware
actor/private projection, exactly-one runtime application, byte hashes,
runtime-bound local physical success, forged unbound traces, incomplete
finger-contact evidence, and timestamp duplication rejection.  It does not
exercise a simulator.
