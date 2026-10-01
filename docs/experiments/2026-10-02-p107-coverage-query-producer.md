# P107 coverage visual-relation query producer

`build_memlite_coverage_visual_relation_queries.py` is the metadata-only
producer for the next diagnostic coverage cohort.  It joins each sealed
selector row to the sealed full-v3 event index by `event_id`, reuses the v5
current-visible-goal templates, and emits one independent query candidate per
supported queried skill.  It does not inspect RGB and does not emit answers,
attempt/recovery outcomes, action payloads, masks, or post-label view IDs.

The producer has a role boundary at the input and output:

* `train` accepts only the sealed `annotation_calibration`/`train` selector
  with its TRAIN queue seal and writes `prelabel_registry.jsonl`.
* `eval` accepts only the sealed `evaluation_only`/`eval` selector without a
  TRAIN queue seal and writes `evaluation_query_registry.jsonl` plus the
  diagnostic-only `evaluation_review.jsonl`.  The latter requires a later
  authenticated `actor_packet.causal_temporal_rgb` packet containing
  `head`, `left_wrist`, and `right_wrist`; offline future frames remain audit
  data and may not be actor evidence.  It is never fed to the TRAIN helper or
  restamped as TRAIN.

Every output candidate is `training_eligible: false` and has
`no_outcome_or_action_labels: true`.  Unsupported canonical skills are
quarantined in `unsupported_goal_queries.jsonl`; they are not rewritten as
action-recognition questions.  A `prelabel_query_id` is the SHA-256 of the
role, event/skill binding, query ordinal/text, and authenticated source pins;
it deliberately excludes answers, evidence, actions, recovery, and any later
canonical view ID.  The compact registry and full `query_candidates.jsonl`
carry the same source-pin object.  A candidate whose pinned taxonomy grounding
is not fully resolved remains in `query_candidates.jsonl` and
`category_grounding_quarantine.jsonl`, but is excluded from every
review-facing TRAIN registry and EVAL review sidecar.  `selected_event_bindings.jsonl`
records eligible, partial, quarantined, unsupported, and explicitly unbound
selected events, so consumers must not assume one query per selected event.

## Planned commands

These commands are the exact metadata-only handoff for the sealed 47 TRAIN +
5 EVAL cohort.  They are documented, not run by this commit; choose fresh
nonexistent output directories before execution.

```bash
python scripts/data/build_memlite_coverage_visual_relation_queries.py \
  --index /home/wsy/behavior-annotations/p107/index-validation/full-v3-candidate-index \
  --selector-root /home/wsy/behavior-annotations/p107/coverage-cohort-next-v1/train \
  --role train \
  --output-dir /home/wsy/behavior-annotations/p107/coverage-cohort-next-v1-queries/train \
  --protocol-path /home/wsy/behavior-annotations/p107/protocol-snapshots/0693b93/src/g05/data/memlite_event_protocol.py \
  --expected-selector-manifest-sha256 f6c2b6f14fa1f3bf0431f9d45c89a68b14bd9a28e96756e542d716c5bde0be4e \
  --expected-selector-selection-seal-sha256 d64202e4acd295d77ac0b362127e13b8a757d33f5861b4fc6485767bdcb3b623 \
  --expected-index-manifest-sha256 351fa44b03fe881b328bbafd5626b986c93879d7f14da346ac8b62a079d200bf \
  --expected-inventory-seal-sha256 7ed1b2c5cbb50c8af042a2b9dca8529a6d133de5fc0fbe224601854235c31479 \
  --category-mapping /home/wsy/behavior-annotations/p107/object-category-grounding-audit/category_mapping.csv \
  --expected-category-mapping-sha256 ef4636716bc1f243f89735ffe89e8e931a2d5739e87164e7146d54d11c681eab

python scripts/data/build_memlite_coverage_visual_relation_queries.py \
  --index /home/wsy/behavior-annotations/p107/index-validation/full-v3-candidate-index \
  --selector-root /home/wsy/behavior-annotations/p107/coverage-cohort-next-v1/eval \
  --role eval \
  --output-dir /home/wsy/behavior-annotations/p107/coverage-cohort-next-v1-queries/eval \
  --protocol-path /home/wsy/behavior-annotations/p107/protocol-snapshots/0693b93/src/g05/data/memlite_event_protocol.py \
  --expected-selector-manifest-sha256 35af755eec72d96adb2f1b3022eb0ce902a6479d016b78ed7a13d94b4076f0d0 \
  --expected-selector-selection-seal-sha256 1dc761c5cbc6f5eece52f9140d88698bc477d8c415863911e046ad4aeb3d277d \
  --expected-index-manifest-sha256 351fa44b03fe881b328bbafd5626b986c93879d7f14da346ac8b62a079d200bf \
  --expected-inventory-seal-sha256 7ed1b2c5cbb50c8af042a2b9dca8529a6d133de5fc0fbe224601854235c31479 \
  --category-mapping /home/wsy/behavior-annotations/p107/object-category-grounding-audit/category_mapping.csv \
  --expected-category-mapping-sha256 ef4636716bc1f243f89735ffe89e8e931a2d5739e87164e7146d54d11c681eab
```

The producer’s defaults pin source release
`90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23`, the
canonical protocol
`7f4f9fbf18fb4ba6ec97a304f0d787ebadb4f84c7184dd041c99027ad84aab0c`, and
coverage expectations
`39ccfb79420010bfc32e2f00d66cae255340a997b20026dc970fdffee412b3c7`.  A
The `--category-mapping` path is required.  It must be the official 2,424-row
taxonomy with SHA
`ef4636716bc1f243f89735ffe89e8e931a2d5739e87164e7146d54d11c681eab`; the
explicit `--expected-category-mapping-sha256` pin is checked before any query
is built.  The inherited v5 resolver keeps an unmapped asset as
`the named metadata entity`/grounding-unknown in actor-facing text and retains
the opaque raw ID only in the audit sidecar.  Missing, malformed, or
wrong-SHA mappings fail closed rather than falling back to suffix heuristics.

The existing phase40 actor adapter expects the older
`p107.visual_relation_query_prelabel_registry.v1` and phase-specific pins, so
these coverage registries are intentionally not silently routed through it.
An independent review helper may consume the rich candidate rows after
checking this manifest and source-pin contract.  The EVAL sidecar is for
direct native-packet diagnostic review only.  No command above creates RGB,
labels, training rows, or release artifacts.

## Review-helper registry contract

The TRAIN registry and the EVAL query registry are JSONL rows with schema
`p107.coverage.visual_relation_query_registry.v1`.  Each row carries exactly
the binding needed for a diagnostic query: `prelabel_query_id`, `event_id`,
`source_group_id`, `query_ordinal_within_event`, `observation_frame`,
`skill_id`, `canonical_verb`, `query_text`, `query_content_sha256`,
`relation_family`, `source_pin`, `usage_role`, `immutable_split`,
`training_eligible:false`, and `no_outcome_or_action_labels:true`.
`source_pin` is the authenticated object containing selector manifest/
selection-seal/job/request and record hashes, full-index manifest/event/
inventory hashes, source release/annotation/group/episode/frame identity, and
the official category-map SHA/row-count/commit.  The registry has no answer,
evidence, recovery, action, or post-label view fields.

The ID is computed as:

```text
sha256(canonical_json({
  "schema_version": "p107.coverage.visual_relation_query.v1",
  "selection_role": role,
  "usage_role": row.usage_role,
  "immutable_split": row.immutable_split,
  "event_id": row.event_id,
  "source_group_id": row.source_group_id,
  "observation_frame": row.observation_frame,
  "query_ordinal_within_event": row.query_ordinal_within_event,
  "skill_id": row.queried_skill_binding.skill_id,
  "canonical_verb": row.queried_skill_binding.canonical_verb,
  "query_text": row.current_visible_goal_relation.query_text,
  "relation_family": row.current_visible_goal_relation.relation_family,
  "goal_scope": "CURRENT_VISIBLE_RELATION_AT_ANCHOR",
  "source_pin": row.source_pin,
}))
```

`canonical_json` uses sorted keys, compact separators, UTF-8, and no NaN.
The EVAL `evaluation_review.jsonl` rows use
`p107.coverage.evaluation_native_review.v1` and additionally require the
native packet field `actor_packet.causal_temporal_rgb`; future audit RGB is
explicitly forbidden as actor evidence.  This producer does not change the
strict phase-v1 actor adapter; a separately reviewed diagnostic reader must
join by `prelabel_query_id` and then verify the source pin before viewing.

The manifest counts distinguish `selected_events`, `candidate_queries`,
`eligible_queries`, `category_grounding_quarantined_queries`, and
`unsupported_goal_queries`; `events_without_eligible_queries` and their IDs
are listed explicitly.  Thus a 47-event selector may legitimately produce
44 eligible TRAIN queries plus three quarantined candidates.
