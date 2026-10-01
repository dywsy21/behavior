# P107 phase40 diagnostic-subset publisher

`scripts/data/publish_memlite_diagnostic_subset_index.py` is a narrow adapter
for the sealed 40-event phase-balanced mini-index.  It writes a new immutable
index; it never modifies the mini-index, query artifacts, annotations, or the
generic label packer.

## Required inputs

The invocation pins all of these before it opens an output staging directory:

- the mini-index inventory seal;
- the full-v3 index inventory seal and manifest SHA;
- the phase selection manifest SHA; and
- the phase queue JSONL SHA.

It requires the queue, selection manifest, mini payloads, group inventory, and
full-v3 lineage to agree on exactly 40 unique event IDs and source groups. All
groups must remain immutable `train` / `annotation_calibration` groups.

The existing phase mini-index uses the legacy
`p107-phase-balanced-mini-index-v1` derivation shape: it seals the parent
event-file SHA but does not contain `parent_source_groups_sha256`. The
publisher accepts that absence only, independently checks the source-group
receipt from the separately inventory-sealed full-v3 manifest, and records the
legacy value as `null` plus the derived full-v3 receipt in its new output. A
present conflicting legacy field is rejected.

The mini and full indexes must pin the same protocol SHA. By default the
publisher reads the repository protocol, but `--protocol-path` can select an
immutable historical producer snapshot only when its bytes equal that shared
pin. The derivative records that reader receipt separately from the new
publisher-script hash, so a historical producer protocol is never presented
as the runtime publisher implementation.

Before staging, the publisher verifies the complete sealed full event file
while retaining only the 40 selected parent events. It cross-checks each
mini-event lineage and queue row against its full-v3 parent: source identity,
parent interval, anchor and phase, queried skill member and bounds, and the
nonfuture actor-causal window. This is strict validation, not a source-index
rewrite.

## Published semantics

The derivative keeps `memlite-event-index-v1` for existing packer input
compatibility, but declares `diagnostic_subset=true` and
`partial_source_coverage=true`. Its coverage report is computed from the 40
copied event rows against the pinned full-v3 task and skill vocabulary:

- present and missing task/skill IDs are explicit;
- `global_vocabulary_complete` is false for the partial sample;
- required task/skill pairs and missing pairs remain `null`; and
- pair coverage status remains `NOT_DECLARED`.

The sealed explicit calibration-selection policy lists exactly the 40 event
and source-group IDs and retains all phase/full-index pins. The output permits
only `annotation_calibration` and fixes training, student, outcome, recovery,
corrective-action, and formal-release fields to false. It is therefore usable
only with the generic packer's existing `annotation_calibration` diagnostic
coverage path; it is not a training or dataset-quality release.

No real derivative has been published by this change. Independent review is
required before running the publisher against the actual phase40 artifacts.
