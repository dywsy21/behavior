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
carry the same source-pin object.

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
  --expected-inventory-seal-sha256 7ed1b2c5cbb50c8af042a2b9dca8529a6d133de5fc0fbe224601854235c31479

python scripts/data/build_memlite_coverage_visual_relation_queries.py \
  --index /home/wsy/behavior-annotations/p107/index-validation/full-v3-candidate-index \
  --selector-root /home/wsy/behavior-annotations/p107/coverage-cohort-next-v1/eval \
  --role eval \
  --output-dir /home/wsy/behavior-annotations/p107/coverage-cohort-next-v1-queries/eval \
  --protocol-path /home/wsy/behavior-annotations/p107/protocol-snapshots/0693b93/src/g05/data/memlite_event_protocol.py \
  --expected-selector-manifest-sha256 35af755eec72d96adb2f1b3022eb0ce902a6479d016b78ed7a13d94b4076f0d0 \
  --expected-selector-selection-seal-sha256 1dc761c5cbc6f5eece52f9140d88698bc477d8c415863911e046ad4aeb3d277d \
  --expected-index-manifest-sha256 351fa44b03fe881b328bbafd5626b986c93879d7f14da346ac8b62a079d200bf \
  --expected-inventory-seal-sha256 7ed1b2c5cbb50c8af042a2b9dca8529a6d133de5fc0fbe224601854235c31479
```

The producer’s defaults pin source release
`90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23`, the
canonical protocol
`7f4f9fbf18fb4ba6ec97a304f0d787ebadb4f84c7184dd041c99027ad84aab0c`, and
coverage expectations
`39ccfb79420010bfc32e2f00d66cae255340a997b20026dc970fdffee412b3c7`.  A
category-mapping file may be supplied with `--category-mapping`; when used,
the inherited v5 resolver authenticates its official 2,424-category SHA and
keeps unresolved names quarantined.

The existing phase40 actor adapter expects the older
`p107.visual_relation_query_prelabel_registry.v1` and phase-specific pins, so
these coverage registries are intentionally not silently routed through it.
An independent review helper may consume the rich candidate rows after
checking this manifest and source-pin contract.  The EVAL sidecar is for
direct native-packet diagnostic review only.  No command above creates RGB,
labels, training rows, or release artifacts.
