# MEM-Lite schema-v6 shared protocol: independent early review

**Review type:** read-only pre-integration contract review.  This is not an implementation approval and does not replace the later data, model-prefix, collate, simulator or end-to-end review.

**Initial reviewed publication (historical baseline):**

- File: `/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py`
- Published and observed SHA-256: `15da12ce9096500873a6b1dfbbb30e690b9ae08c33e69ac8b9750daae954677f`
- Interpreter: `/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python`
- Import origin: exactly the published data-tree file above; no `activate_g05.sh` was sourced.
- `py_compile`: passed.

## Historical intermediate recheck: candidate SHA `1305d3aac3d6f62dcf142e5075e9888fc2b686b84c1d17d3898065bd657d013d`

The exact absolute-interpreter import was rechecked against the data-tree candidate. The original ten P1 counterexamples are now rejected: unknown input evidence is strictly past, booleans are strict, supervised outcomes need a prior evaluated bundle, past action horizons and inconsistent decision/bundle rows are rejected, v5 conflicts are rejected, and unknown low skills need `low_action_supervision_mask=false`. The writer's targeted test file is `tests/test_memlite_skill_protocol_v6.py`; it is distinct from the initial generic test path used by this review.

The candidate was **not an approval**. One deployment P1 boundary remained:

1. Its top-level semantic projections remove `bundle_id`, but an `UNKNOWN` binding causes [semantic_active_skills](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:298>) to put the entire `raw_relation` tree in model text. An independent relation containing `bundle_id`, `episode_index` and nested `provenance.source_id` appears verbatim in `active_skills_text`. Complete relation provenance may remain in the audit record; deployed semantic text must recursively remove/route source and audit IDs.

Parallel completeness is an **offline E1 label-generation gate**, not a deployment parser oracle. Runtime may select any legal current singleton or multi-skill bundle; the parser should validate only legality, uniqueness and stable semantic ordering. In contrast, an offline training label must compare its emitted active members against an independent expected-member set derived from the original annotation time coverage. [bundle_member_keys_json](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:461>) is optional and otherwise synthesized from the supplied subset, so it cannot by itself prove source completeness. The generator needs source group identity plus expected members/cardinality; these remain audit-only and must be excluded from every model prefix/text.

Parent-goal semantics need a separate correction, not a blanket removal. In the candidate, the singular `parent_goal` is still an input field and no target/new parent-goal field appears in the planner target projection. The required eventual contract is: stable `task_name`; causal `previous_parent_goal` as high input; `target_parent_goal` as high-output supervision and low expert condition; a primitive-unbound record masks **that supervision only**. A primitive-derived parent goal is a legitimate predicted command/plan, not an already-observed future fact. `target_parent_goal` must therefore not enter the high input prefix.

The reproducible, read-only probe is [memlite_skill_protocol_review_probe.py](/home/wsy/behavior/memlite_skill_protocol_review_probe.py).  It is copied into the integration-only source tree at `scripts/memlite_skill_protocol_review_probe.py`; it must be run against a published module root and never imported by production code.

## Initial P1 findings — all rechecked as fixed in the final candidate below

1. **UNKNOWN planner input can carry future evidence.** At [memlite_skill_protocol.py:280](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:280>)–288, `known_previous_outcome_evidence_end_frame` is checked only when `known_previous_outcome != UNKNOWN`. Counterexample: `frame_index=5`, `known_previous_outcome=UNKNOWN`, `known_previous_outcome_evidence_end_frame=99` is accepted. UNKNOWN needs either `-1` or a past/current, auditable evidence boundary; it must never carry a future frame.

2. **Python truthiness silently changes labels.** At [line 291](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:291>) and [line 341](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:341>), `bool(...)` accepts strings. `outcome_supervision_mask="false"` becomes `True` and accepts a supervised SUCCEEDED target; `task_complete="false"` with STOP becomes `True`. Require an actual boolean or one explicitly parsed, unambiguous serialized representation.

3. **A supervised success/failure has no required evaluated skill.** At [lines 295–305](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:295>), a masked FAILED/SUCCEEDED event with valid evidence but `evaluated_skills_json="[]"` is accepted. This makes the outcome target untraceable to a previously executed bundle.

4. **Action and decision time semantics admit impossible rows.** At [lines 338–357](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:338>), `frame_index=15` with active skill end/action horizon 10 is accepted; the code never checks that the low action target is in the future. It also accepts high empty bundle + EXECUTE, low nonempty bundle + STOP, and task_complete + active low action. Contract-level decision/bundle constraints are needed before a loader converts them to actions.

5. **v5 and v6 can contradict, and no projection is enforced.** [Line 329](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:329>) retains every input field through `dict(row)`, while [lines 54–73](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:54>) merely declare field-name tuples. `status=DONE`, `task_complete=false`, `next_decision=EXECUTE` is accepted. A valid record also retains `outcome_target`, outcome evidence and an arbitrary future/audit field. Keeping these fields in an audit record is not itself a leak, but consumers currently have no compulsory projection. Define a v5↔v6 policy (reject mixed conflicts or explicit translation) and test an allow-list projection at each planner prefix, outcome head input and collate boundary.

6. **Unknown skill can become a low-level training condition.** [Line 42](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:42>) permits `SKILL_UNKNOWN`; [lines 317–347](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:317>) contain no active-skill supervision mask or low-branch prohibition. A low row with raw `skill_id=999` is accepted. Preserve provenance but prevent it from becoming FM action supervision unless an explicit mask/approved fallback exists.

## Initial P2 findings — disposition recorded by final recheck below

1. **Relation tree loses information.** [Lines 140–157](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:140>) discard `destination`, `relation` and provenance inside `raw_relation`, despite the comment saying extra provenance is retained. [Lines 179–188](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:179>) also accept BOUND PLACE_IN with empty target and destination. Specify per-verb required relation roles and retain a canonical structured relation/audit representation.

2. **Parallel bundle completeness needs an external label-generator check.** A deployment parser may accept any legal singleton or multi-skill bundle; it must not infer an oracle expected count. Offline training labels, however, must compare emitted members with an independent expected-member set from original annotation time coverage. Source group identity, expected membership/cardinality and the comparison result are audit-only; they must not enter model text/prefix.

3. **Text condition delimiters are unescaped.** [Lines 250–264](</mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/src/g05/utils/memlite_skill_protocol.py:250>) interpolate target/source/destination raw text into `;`/`=` separated fields. A target named `plate; arm=LEFT` creates an ambiguous condition. Use encoded scalar fields or reject control delimiters.

## Initial evidence that passed

The probe and direct canonical-input checks verified:

- Default original-demo `UNKNOWN` with `outcome_supervision_mask=false` is accepted.
- A real, evidenced `UNKNOWN` (`mask=true`, past evidence, non-missing evidence kind) is accepted; this is the intended distinction from an unobserved demo.
- A two-leaf parallel bundle is preserved without deduplication.
- Low empty bundle, UNKNOWN binding with a claimed target, supervised future outcome evidence, known SUCCEEDED future evidence, noncanonical JSON order, and `task_complete=true` + EXECUTE are all rejected.

## Final targeted recheck: candidate SHA `48d6f6d263b2891fdbf6e837a86f2a64ee79effb8a1268e41bfd6368e5ed12a0`

**Result: the shared protocol/data-builder contract is approved only as an E1 input candidate; it is not a model-training or success-rate approval.**  The exact data-tree protocol file matched this SHA-256 and was imported with `/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python` without sourcing `activate_g05.sh`.

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$DATA/src:$DATA" ... pytest -q -p no:cacheprovider tests/test_memlite_skill_protocol_v6.py tests/test_memlite_skill_builder_v6.py` passed **19/19**.
- The independent probe at [memlite_skill_protocol_review_probe.py](/home/wsy/behavior/memlite_skill_protocol_review_probe.py), run with `--require-fixed`, rejected all ten initial counterexamples.  It also confirmed that nested `bundle_id`/episode/source audit identities do not enter `active_skills_text`, while the audit record retains its fields.
- The builder independently recomputes each offline bundle's expected member keys from raw leaf interval coverage, checks that against the event accumulator, and records the expected set as audit-only.  The deployment parser still permissibly accepts a valid singleton current bundle; it does not use an oracle expected member count.
- The parent-plan contract is now explicit: `task_name`, `previous_parent_goal`, memory, previous intent, known prior outcome and execution feedback are high-prefix inputs; `target_parent_goal` is EOC-after high supervision and `parent_goal` is the low-level expert condition.  The builder writes natural-language `task_name` on every row; tests confirm it is present in high input projection and `target_parent_goal` is absent.
- A primitive that cannot be bound preserves its leaf action data, masks only `parent_goal_supervision_mask`, and does not claim a physical completion fact.

**Still outside this review / blocking training:** the model tree is still pinned to old protocol SHA `15da12ce...`, so its builders and templates have not yet proven the final semantic projection.  The training entrypoint also lacks the required stage lifecycle hook (checkpoint remap → base load → LoRA injection/restore → stage trainability gate → DDP/optimizer).  These are independent P1 model/loader blockers; do not start a GPU run from this result.

## Completed review checklist

1. **Done:** re-ran the probe against exact SHA `48d6...12a0` with `--require-fixed`.
2. **Done:** rechecked strict booleans, causal UNKNOWN, outcome-to-prior-bundle linkage, action horizon and decision/bundle constraints.
3. **Done at protocol level:** v5 conflict fields are rejected by the v6 validator.
4. **Still blocking at consumer level:** audit records may retain identifiers, but the low builder, high builder, collate and runtime prefix must each prove their recursive semantic allow-list after synchronizing this final protocol.  The data protocol's own projections pass; the model tree has not yet been rechecked.
5. **Done for E1 generator / still blocking consumers:** the builder records independently derived expected members and model text excludes audit identity; later consumer/collate review must prove those audit fields cannot re-enter tokens.
