# P107 actor-query data-prep contract

This note describes the additive plumbing in
`src/g05/data/memlite_event_actor_query.py`. It is a preparation-data
contract, not a training authorization or a model-input/collator contract.

## Optional sealed registration

An immutable release may add this manifest key:

```json
"actor_query_sidecar": {
  "schema_version": "p107-actor-query-sidecar-v1",
  "path": "actor_queries.jsonl",
  "sha256": "<sidecar bytes SHA-256>",
  "row_count": 0,
  "prelabel_query_registry": null
}
```

When a goal query is present, `prelabel_query_registry` is required and has
the exact fields `schema_version`, `path`, `sha256`, and `row_count`. Both
declared paths must also have matching entries in the release `files`/seal
receipts. A sidecar row is exactly:

```json
{"schema_version":"p107-actor-query-sidecar-v1",
 "view_id":"<postlabel view SHA-256>",
 "actor_query":{...}}
```

The prelabel registry row is exactly `schema_version`, `prelabel_query_id`,
`query_content_sha256`, `text`, `source_event_id`, and `source_skill_id`.
`query_content_sha256` hashes only kind/text. `prelabel_query_id` hashes the
prelabel schema, source event, source skill, and content hash. These identities
are deliberately distinct from the postlabel `view_id`; projection also
requires the registry source event to equal the bound view event. The packer
must bind a query supplied by the frozen prelabel registry; it must not
reconstruct query text from a postlabel `goal_relation`.

## Query variants

The validator accepts only these exact structured fields:

- `goal_satisfaction_counterfactual`: `kind`, `prelabel_query_id`,
  `query_content_sha256`, `text`. The desired/counterfactual question may
  condition a future goal head; its answer and visual evidence remain target
  fields. Registry rows are content-hash checked.
- `attempt_outcome_intent`: `kind`, `attempt_id`, `issued_frame`,
  `query_content_sha256`, `text`. Dataset projection binds the attempt ID to
  the actual attempt view and requires `issued_frame <= observation_frame`.
- `runtime_high_level_intent`: `kind`, `intent_id`, `issued_frame`,
  `query_content_sha256`, `text`. It is the only deployable variant and must
  be issued causally by the high-level runtime planner.

Structured answer/result/review/evidence/privileged/object-instance/segment-
end/future fields are rejected. Query text is bounded and hashed but is not
word-filtered; legitimate text such as “completed” is allowed. For a
low-action-supervision corrective view, only a runtime intent whose ID matches
the executed action-intent bundle is accepted. A counterfactual query never
turns an expert action into BC supervision.

## Compatibility boundary

Packages without `actor_query_sidecar` retain the current projected row
exactly. Existing protocol-pinned and full-v3/40 packages are not rewritten or
rehashed. `for_training=True` remains fail-closed, calibration roles remain
non-student, and no optimizer or collator is changed by this plumbing.
