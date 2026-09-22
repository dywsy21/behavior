# H09AA — separate completion-state supervision candidate

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
