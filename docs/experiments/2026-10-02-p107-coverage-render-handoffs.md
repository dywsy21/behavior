# P107 coverage render handoffs (metadata-only)

This adapter turns a sealed diagnostic coverage selection into a renderer
input without changing source roles or manufacturing labels. It does not
decode RGB, infer outcomes, or make any candidate training-eligible.

## TRAIN

`publish_memlite_coverage_render_handoff.py` verifies the selector manifest,
selection seal, queue seal, full-index identity, inventory, coverage, and
protocol pins. It writes a fresh subset index and invokes the existing queue
builder, then validates the resulting canonical queue seal and resume path.

The canonical `train/` directory contains only the existing queue payload
inventory and `camera_native_render_requests.jsonl`; selector reports and
prior-window provenance are kept in the sibling
`train_coverage_provenance.json`. A typical invocation is:

```sh
python scripts/data/publish_memlite_coverage_render_handoff.py \
  --index /path/to/sealed-index \
  --train-selector /path/to/coverage/train \
  --output-root /path/to/fresh-cohort/train-handoff \
  --coverage-expectations /path/to/coverage-expectations.json \
  --protocol-path /path/to/pinned/memlite_event_protocol.py \
  --expected-protocol-sha256 PINNED_PROTOCOL_SHA256 \
  --expected-source-release-manifest-sha256 PINNED_RELEASE_SHA256 \
  --expected-inventory-seal-sha256 PINNED_INVENTORY_SHA256 \
  --expected-coverage-expectations-sha256 PINNED_COVERAGE_SHA256 \
  --expected-selector-manifest-sha256 SELECTOR_MANIFEST_SHA256 \
  --expected-selector-selection-seal-sha256 SELECTOR_SELECTION_SEAL_SHA256 \
  --expected-selector-queue-seal-sha256 SELECTOR_QUEUE_SEAL_SHA256
```

## EVAL

`render_memlite_evaluation_packets.py` accepts only an `evaluation_only` /
`eval` selector handoff. It uses the shared packet assembly backend with an
explicit evaluation role and emits `evaluation_render_receipt.json` using
schema `p107-evaluation-render-authorization-v1`. Its exact fields are:

```text
schema_version, status, training_eligible, usage_role, immutable_split,
selector_manifest_sha256, selector_selection_seal_sha256,
selector_request_filename, selector_request_sha256, index_manifest_sha256,
index_event_file_sha256, inventory_seal_sha256, source_release_manifest_sha256,
canonical_protocol_sha256, packet_manifest_sha256,
packet_manifest_queue_authority_sha256, packet_count, request_bindings,
no_outcome_or_action_labels
```

The receipt has `status=EVALUATION_ONLY_RENDER_RECEIPT`,
`training_eligible=false`, `usage_role=evaluation_only`, and
`immutable_split=eval`. `packet_manifest_queue_authority_sha256` equals the
evaluation selector selection-seal SHA; this is an evaluation authority, not
a TRAIN queue seal. The EVAL output never contains `queue_seal.json` and
cannot enter the TRAIN queue or BC publisher.

A diagnostic EVAL invocation is:

```sh
python scripts/data/render_memlite_evaluation_packets.py \
  --index /path/to/sealed-index \
  --selector-output /path/to/coverage/eval \
  --output /path/to/fresh-cohort/eval-packets \
  --coverage-expectations /path/to/coverage-expectations.json \
  --protocol-path /path/to/pinned/memlite_event_protocol.py \
  --expected-protocol-sha256 PINNED_PROTOCOL_SHA256 \
  --expected-source-release-manifest-sha256 PINNED_RELEASE_SHA256 \
  --expected-inventory-seal-sha256 PINNED_INVENTORY_SHA256 \
  --expected-coverage-expectations-sha256 PINNED_COVERAGE_SHA256 \
  --expected-selector-manifest-sha256 EVAL_SELECTOR_MANIFEST_SHA256 \
  --expected-selector-selection-seal-sha256 EVAL_SELECTION_SEAL_SHA256
```

The default ten-slot temporal schedule is
`[-60,-45,-30,-15,0,1,16,31,46,60]`. Frames at or before the observation
anchor are causal actor evidence; later frames are offline audit evidence and
must never be supplied as earlier actor input. The selector's `BOUNDARY`
stratum means an event/episode boundary only; it is not proof of a skill
boundary, outcome, recovery, or physical success.

Production commands were not run for this change. The focused tests use
sealed temporary metadata fixtures, `decode=False`, and verify TRAIN queue
resume/seal validation, EVAL receipt role gates, protocol/request tamper
failure, and the existing renderer regressions.
