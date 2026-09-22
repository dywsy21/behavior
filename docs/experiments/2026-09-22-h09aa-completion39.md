# H09AA — separate completion-state supervision candidate

## Actual result, 2026-09-22 11:27:40 BJT

The CPU block completed in1024s from11:10:36, before11:30:36. Fixed source
`ccb16f2ee268504c9bffde37179b4a06abcdf6e4` includes parent-requested fail-closed
checks for exactly zero base velocity in the cleanup HOLD and identical
physical-frame/verdict ticks. The original6f4e9a6 candidate remains unchanged.
New source is a separate clean Git worktree on robo; public executor digest
remains8fdfcd3e1b8bdfd95530eec09710d1c44efc215885b49c64c1a82f55bbc501a8.

Local208 SFT tests/6.244s and420 public tests/7.589s passed. Formal robo3.10
passed208/13.674s, then rebuilt all five whole-run admissions and39 state rows
in24.991s. All original34 motion rows and per-run receipts match exactly;
new and old39 sidecar bytes and10-state review lists are also identical.
No physical reset/control, model call, GPU training update or runtime alias.
Results-root size after construction1,340,191,996B remains under6GiB.

Full candidate package is local at own artifacts
`h09y-resume-20260921/completion39_ccb16f2_v1` and remote at
`/mnt/nvme_tmp/robodojo_vlm_sft_20260919/h09y_grasp_only/completion39_ccb16f2_v1`:

- `states.jsonl`:2fd27d707ad86e7fd6c09662000e99af26c0b1d686bbe9ff92b3c6aa8604fa99.
- `manual_review.json`:005a8b0cb7d08da01054f3b559931f99419f769a3ca9a7d4d696ab346665b4d6.
- `completion_dataset.json`:085d0a06eb590d0f390337951ca47ba412009010e69cd51a8665d838e8aba0fd.
- `author_validation.json`:ccb65c7fd30a8e361df0f30c1c54d5a5041679215b717b2ccd76bd590113ca9a.

The adjacent `completion39_ccb16f2_manual_local_paths.json`, SHA
6b1338d75ff5c4f875966b5244da440c3243f690617af622279289fac408e419,
maps all10 captures/30 original RGBs to existing local archives and verifies
every capture/image hash, including W396's older archive. No whole trajectory
was recopied for this review. Parent independent code/data/manual review is
pending; status remains CANDIDATE_PENDING_PARENT_REVIEW_NO_TRAINING.

## Registered design and scope

Owner: Astra-max-vlm_sft_resume_20260921. CPU-only block began2026-09-22
11:10:36 BJT, hard deadline11:30:36. No new physical reset, model call,
training, runtime early stop or modification of the original120-update adapter.

The original five fully reviewed TRAIN trajectories provide34 pre-macro
IN_PROGRESS observations and five captured, stable successful terminal states.
The new builder binds exactly dataset5b2b5ee0/rows ef1a62fa/source-list5e5a91b8,
replays the existing whole-run admission under formal roboPython3.10, and
requires every original motion row and per-run receipt to remain exactly equal.
It checks current capture bytes/q/FK/clock, actual completed history, the
same-clock outcome, and the following real one-control pose/grip-preserving
cleanup HOLD. That single control is not a demonstration of an18-control
HOLD primitive. Unknown/failed outcomes cannot become CONTINUE labels.

The separate versioned output is `{skill_status, motion}`. CONTINUE requires
one unchanged native45 symbol and keeps the original motion loss; REQUEST_VERIFY
requires null motion and masks motion loss. A request is not a local or official
success assertion. The six-field public actor and existing public text/images
stay unchanged. All private outcomes/reviews are offline label provenance,
never actor inputs. No i1/i71 observations or labels enter these sidecars.

The saved text is only the original public-input record: its old motion-only
output instruction is not suitable for training completion JSON directly.
A separately reviewed versioned output formatter is required in a future
training-integration ticket; it is deliberately absent from this data block.

Outputs are39 candidate state rows, a10-state three-view manual-review list
(each trajectory's final before-macro negative and terminal positive), and a
hash-bound manifest marked CANDIDATE_PENDING_PARENT_REVIEW_NO_TRAINING.
Original dataset/quarantines/physical evidence remain untouched. This block
does not implement a runtime verifier/handoff, change any success predicate,
or authorize a new training run. A separate source/data/manual parent review
and explicit integration ticket are mandatory before subsequent use.

Scope limitation: five GRASP terminal examples do not establish a general
cross-task completion detector. Existing i71 FT and proprio/history NN both
reached transient local success and later collided; neither strict terminal
outcome is relabeled, and these observations do not establish a visual advantage.
