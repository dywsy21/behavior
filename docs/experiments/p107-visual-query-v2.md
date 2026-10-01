# P107 phase40 prelabel visual-query producer

This producer creates metadata-conditioned query candidates only.  It does not
decode RGB, attach evidence, answer a query, create an annotation, emit action
supervision, or assign a postlabel canonical view ID.

## Input and identity contract

Run `scripts/data/build_memlite_visual_relation_queries.py` with the sealed
phase queue, the exact `phase_candidate_index/event_candidates.jsonl`, and the
sealed `phase_selection_manifest.json`.  The producer verifies the manifest's
queue/index hashes and 40-row budget, then joins by `event_id`.  For every row
it checks source group, task, anchor frame, parent event, phase stratum, queried
skill ID/description/canonical verb, skill start/end, target/source/destination/
target-part, and original segment interval.  Positional or selection-order
joins are rejected.

Each query has a `prelabel_query_id` computed as

```text
sha256(canonical_json(metadata/query identity))
```

The identity contains the event/skill/phase binding, exact atomic query text,
role-conditioned metadata entities, and immutable source-record hashes.  It
does not contain an answer, RGB evidence, action field, recovery decision, or
postlabel view ID.  `query_ordinal_within_event` is part of the identity, so
multiple queries for one event cannot overwrite one another (the q24-style
regression is covered by the focused test).

## Query semantics

- The primary question is `CURRENT_VISIBLE_RELATION_AT_ANCHOR`: one atomic
  visible relation, with exact metadata IDs and explicit grounding gates.
- `historical_change_relation` is separate and requires the same entity in an
  observable prestate and poststate.  Segment end, gripper closure, timeout,
  contact, or an endpoint window does not prove a state change or transfer.
- `PRESS` asks only for contact with an identifiable control; contact with an
  object body is not a press result.
- `HANDOVER` accepts an explicitly grounded receiving agent or other robot
  gripper; it never invents a person recipient.
- `NAVIGATE` retains required metric distance and pose.  Target visibility or
  apparent proximity is a diagnostic, not goal supervision, without approved
  state evidence.
- Geometry is current geometry only.  Opening/closing/placement orientation
  and the historical state transition are separate scopes.
- Object/location/reference roles remain unresolved when metadata does not
  explicitly bind them; the producer does not turn every `object_id` into a
  destination.
- Only causal frame indices at or before the anchor are retained in the actor
  policy.  Later frames are marked offline-review-only and are not emitted as
  actor evidence references.

## Output

The external output directory contains `query_candidates.jsonl`,
`prelabel_registry.jsonl`, `build_metadata.json`, and `manifest.json`.  It is
calibration-only (`training_eligible=false`) and records truthful provenance:
gpt-5.6-luna/max producer, `human_reviewed=false`, `image_inspected=false`,
and `labels_created=false`.

