# Storage evidence

This directory contains a small, text-only evidence bundle for the 2026-09-12
storage operation. It records the verified cold `experiments` archive and the
subsequent link switch. It does not contain model weights, arrays, videos,
runtime environments, raw process listings, private command lines, or large
logs.

## Final layout

- The server path `/mnt/sdc1/robodojo/experiments` is now a symlink whose exact
  target is `/mnt/tmp1/robodojo-archive-20260912/experiments`.
- The temporary link and the exact isolated source directory
  `experiments.archived-source-20260912` are absent after the authorized
  duplicate removal. No other source directory was removed.
- The archive remains a real directory with 353 regular files and 126
  directories; its metadata matches the pre-migration manifest, with zero
  symlinks in the archived tree. Representative `model.safetensors` files
  remain readable through the archive link.
- The Hugging Face source and target caches were explicitly outside this
  operation and remain in place.

## Content and operation receipts

Content verification used:

```text
rsync -aHAXS --numeric-ids --checksum --dry-run --itemize-changes --stats
```

The receipt reports `exit_code=0`, `itemized_change_count=0`, 353 regular
files, 126 directories, and equal source/target allocated bytes of
`516608692224`. Per-file SHA-256 values were intentionally not computed;
`rsync --checksum` is the recorded content-verification method and must not be
described as a per-file SHA receipt.

The source removal was limited to the exact, already-verified duplicate
directory. It used `find` with `-xdev`, `-depth`, and an exact root followed by
`rmdir`; no recursive broad-root deletion was used. The archive was retained.

The canonical receipt values are:

```text
source available before: 302917308416 B
source available after:  819505172480 B
net release:             516587864064 B (0.516587864064 TB; 0.469833925366 TiB)
tmp1 available after:    113477279744 B
```

An independent postcheck at `2026-09-12T14:26:46Z` observed current free
space of `819444875264 B` on `/mnt/sdc1` and `113477316608 B` on `/mnt/tmp1`.
The source value can decrease while the active evaluation writes new
`behavior_dev` outputs; this does not change the migration receipt.

## Runtime safety postcheck

At the final check, task2 PID `3632286` was still running and the seven
loopback service ports `8772`, `8773`, `8776`, `8777`, `8778`, `8780`, and
`8781` were still listening. No experiment archive files were open after the
switch/removal check.

## Recovery boundary

Use [experiments_recovery_plan_20260912.txt](experiments_recovery_plan_20260912.txt)
for the future recovery procedure. It requires staging the archive into a new
real directory on `/mnt/sdc1`, running the same checksum dry-run, confirming
the source path is unused, and only then switching the path. Do not rsync into
an existing `experiments` symlink. Keep the archive until an independently
verified rollback is no longer needed.

## Evidence files

The copied receipts and manifests are small text artifacts from
`/mnt/tmp1/robodojo-archive-20260912/manifests/`:

- `experiments_checksum_receipt_20260912.txt` — rsync content-verification
  method and exit status.
- `experiments_migration_final_receipt_20260912.txt` — final path, counts,
  free-space release, and untouched HF boundary.
- `experiments_migration_postswitch_receipt_20260912.txt` — link and backup
  checks before isolated-source removal.
- `experiments_source_removal_receipt_20260912.txt` and
  `experiments_source_removal_runtime_20260912.txt` — exact removal scope
  and runtime guards.
- `experiments_directory_metadata.tsv`,
  `experiments_target_files_metadata_final.tsv`, and
  `experiments_source_removal_manifest.tsv` — filename/size/type metadata,
  not payload data.
- `verification_artifact_checksums.sha256` — server-side hashes for the
  selected receipts and related verification artifacts; it is not a payload
  checksum list. It contains absolute `robo` paths and must be checked on
  `robo`; do not run it locally as though it were a complete local-bundle
  verification list.

The source archive is not vendored into Git. These files preserve provenance
without putting the 516 GB experiment payload into the collaboration repo.
