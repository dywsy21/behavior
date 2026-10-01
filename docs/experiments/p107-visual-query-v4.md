# P107 phase40 visual-query producer v4

This revision keeps the public prelabel registry schema at
`p107.visual_relation_query_prelabel.v1` while grounding prompt nouns with an
explicit external input:

- BEHAVIOR-1K BDDL commit: `bd049de3119acdcdf2334fe9e1ebe060fa20c108`
- file: `bddl3/bddl/generated_data/category_mapping.csv`
- expected SHA-256: `ef4636716bc1f243f89735ffe89e8e931a2d5739e87164e7146d54d11c681eab`
- expected unique category rows: `2424`

The build accepts this CSV through `--category-mapping`; it is not copied into
the repository.  For each raw metadata ID, the resolver selects the unique
longest official category prefix followed by `_`, and renders that category
with underscores replaced by spaces.  The raw ID, opaque suffix, synset, and
taxonomy pins are retained only in `category_grounding_audit`.  No fixed-width
suffix parser is used.  No-match, tied, or empty-suffix inputs are marked
`UNKNOWN_CATEGORY` and quarantine the candidate rather than receiving a
guessed noun.

Role ambiguity remains independent of category wording: target, source,
destination, material, and reference roles still require RGB grounding.  The
query producer emits no answers, labels, action payloads, outcome claims,
future actor evidence, or postlabel view IDs.  Navigation distance/pose,
press success, handover recipient identity, and historical state changes keep
their existing state/prestate gates.

Reproduce the external v4 output with:

```sh
python scripts/data/build_memlite_visual_relation_queries.py \
  --queue /home/wsy/behavior-annotations/p107/phase-balanced-calibration40-v2/phase_balanced_queue.jsonl \
  --phase-index /home/wsy/behavior-annotations/p107/phase-balanced-calibration40-v2/phase_candidate_index/event_candidates.jsonl \
  --selection-manifest /home/wsy/behavior-annotations/p107/phase-balanced-calibration40-v2/phase_selection_manifest.json \
  --category-mapping /home/wsy/behavior-annotations/p107/object-category-grounding-audit/category_mapping.csv \
  --output-dir /home/wsy/behavior-annotations/p107/phase-balanced-calibration40-v4-questions
```

The v4 output remains calibration/review-only (`training_eligible:false`),
with agent provenance `gpt-5.6-luna/max`, `human_reviewed:false`, and
`root_review:PENDING`.
