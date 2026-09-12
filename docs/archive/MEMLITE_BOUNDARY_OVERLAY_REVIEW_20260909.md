# MEM-Lite v6 boundary/quarantine overlay — independent review

Date: 2026-09-09  
Reviewer: coordination integration

## Scope and verdict

**PASS_FOR_PENDING_COMPOSITE_RELEASE only.** This is a code/artifact integrity review of a metadata-only action-boundary overlay. It is not a human data-release signature, formal-training authorization, outcome/physics proof, task-success result, or permission to modify the currently frozen Stage-A source tree.

The reviewed candidate is bound to the immutable clean-r2 labels SHA-256 `666f8fc0b097ad963de7c3e484d4017c59e8d0dfc74eaaa6124d6c551ac31cc6`:

- candidate root: `/mnt/sdc1/robodojo/datasets/memlite_skill_annotations_task0_4_v6_boundary_quarantine_overlay_candidate_20260909`
- manifest: `meta/manifest.json`, SHA-256 `d4a0cb21f9380d85d86eaf20f02284a961f8933c15d47bda6885525e25239404`
- boundary ranges: `meta/boundary_ranges.parquet`, SHA-256 `51ca4b58dbdcb58eedcfeb9926f261e99840fe800c5bf2562c2b8c4d006bae85`
- quarantine ranges: `meta/quarantine_ranges.parquet`, SHA-256 `94e6d4d65ab93a797c3e15b4001523055f067b8acbb83996c3eecdd08dfb5684`

The candidate must remain `PENDING_INDEPENDENT_REVIEW` / `review_passed=false` until the separate immutable root-decision and reviewed-composite release record are issued. That record must pin all three overlay bytes—not merely its eligible digest—plus r2 labels, reader/projection code, the human-review receipt and exclusions.

## What was checked

- `MemLiteBoundaryOverlay` verifies the base labels and both parquet artifact hashes before use, rejects malformed/overlapping intervals, uses half-open ranges, and fails closed for a low row without an exact membership range.
- It can only reduce a raw low action/segment horizon; it rejects any effective boundary after either immutable raw boundary. It changes no model semantic projection and retains raw/effective boundaries only in audit-side records.
- Sampling subtracts exactly the explicit quarantines. Branch-qualified direct draws of excluded tuples are required to fail rather than resample.
- Candidate counts are 17,660 low membership ranges, 9,153,344 valid low rows, 312 metadata-shortened horizons, 62 altered 32-step masks, and exact quarantine counts low=271/high=2.
- Direct artifact read confirmed: ep821/f7995 and f7996 low end at 7997; ep821/f7997 and f8267 are quarantined in both branches; ep892/f8044 ends at 8045. The candidate reports no raw-audit projection leak.

Reviewed code identities:

- `src/g05/data/memlite_boundary_overlay.py`: `0d20b1ce7cf15521091d7bc0fed1e080069abdb2239c83f66d4a648dd6ce3d00`
- `src/g05/data/base_lerobot_dataset.py`: `ae2a6f1a692af10ca054ecedc17ea0b46767a21b26b41ff7924d9c1e9d5f3f8c`
- `src/g05/data/memlite_audit_resolver.py`: `8e77dfe3615710cf6d10e6f69ec0410f5d7dee5e4f9906a9e5ef499a7e27c046`
- generator: `scripts/data/build_memlite_v6_boundary_quarantine_overlay.py` = `242f4f2d5405c65edf871f37f234f5dc7ce1e2ea965130d0683cb78a061f5993`
- full mechanical verifier: `scripts/data/verify_memlite_v6_boundary_overlay_artifact.py` = `3500fc843ccd78337fe848089880bc0f58129f787eb1f397abbba4c69c2b1aaa`

Independent targeted CPU run used the approved venv, `PYTHONDONTWRITEBYTECODE=1`, source-only import path and no pytest cache. It passed 49 tests (one known pynvml deprecation warning): boundary overlay, sidecar overlay, v6 protocol, strict relation projection and low empty-target processor tests.

Relevant existing receipts remain evidence, not replacement for this review:

- full mechanical validation: `memlite_v6_boundary_overlay_full_mechanical_validation_20260909.json`, SHA-256 `477b799233c42fdda380fdcfc76255b2572992904d25560126b26b9d20c1db8d`
- combined train join/exact-reject: `memlite_v6_boundary_overlay_train_join_and_exact_reject_20260909.json`, SHA-256 `67e7a9bfe1938de7ba47f445b36ac45788b2beb234f3a955af96e196f7466f8d`
- final-code ep821/f7996 actual tuple: `memlite_v6_boundary_overlay_ep821_f7996_actual_finalcode_20260909.json`, SHA-256 `22185d76531a5859288c7021c222e6f2f08e3a5b1068a52323f727b3c5061614`

## Release conditions retained

The root decision must still preserve R2 byte immutability, retain the 271 low plus 2 high exclusions, and keep uncertain visual/outcome claims as `UNKNOWN` with the appropriate supervision masks. The new composite loader/config hook must invalidate train/resume identity if the overlay manifest, ranges or quarantine bytes change. This review does not erase outstanding human-review coverage, R1→R2 migration evidence, formal Stage-A smoke/resume gates, or E7 serving/evaluation work.
