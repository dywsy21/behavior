# P107 causal review-page builder

`scripts/data/build_memlite_causal_review_pages.py` creates deterministic
native-pixel helper pages from a sealed MEM-Lite packet directory.  The
default output is actor-facing and uses only
`actor_packet.causal_temporal_rgb`; every selected sample must have role
`ACTOR_CAUSAL` and absolute `sample_frame <= observation_frame`.  The builder
also checks the packet schedule, all three camera views, receipt SHA-256,
PNG/RGB mode, and native dimensions (head `720x720`, wrists `480x480`).

Use explicit source bindings; no adjacent queue is inferred:

```bash
python3 scripts/data/build_memlite_causal_review_pages.py \
  --sealed-root /path/to/sealed-packets \
  --index-root /path/to/event-index \
  --queue-path /path/to/selection_queue.jsonl \
  --queue-seal-path /path/to/phase_queue_seal.json \
  --expected-queue-seal-sha256 QUEUE_SEAL_SHA256 \
  --expected-packet-manifest-sha256 PACKET_MANIFEST_SHA256 \
  --output /path/to/new-causal-review-pages \
  [--query-registry /path/to/query_candidates.jsonl \
   --expected-query-registry-sha256 QUERY_REGISTRY_SHA256] \
  [--coverage-event-bindings /path/to/selected_event_bindings.jsonl \
   --expected-coverage-event-bindings-sha256 EVENT_BINDINGS_SHA256 \
   --coverage-provenance /path/to/train_coverage_provenance.json \
   --expected-coverage-provenance-sha256 PROVENANCE_SHA256 \
   --coverage-selector-root /path/to/original/train-selector] \
  [--render-requests /path/to/camera_native_render_requests.jsonl \
   --expected-render-requests-sha256 RENDER_REQUESTS_SHA256] \
  [--audit-output /path/to/new-future-audit-pages]
```

The packet manifest and phase queue seal are externally byte-pinned.  The
builder verifies their exact file inventories and receipts, binds the queue
and mini-index payloads to the seal, and cross-checks source-release and
protocol digests through the queue, index, event, source-group, and packet
records.  It does not infer a seal from a neighboring directory.

Canonical metadata-queue TRAIN packets may instead be bound to the standard
`p107-metadata-annotation-queue-seal-v1` handoff.  When that authenticated
packet manifest claims `render_requests_sha256`, `--render-requests` is
required and must be the exact sealed
`camera_native_render_requests.jsonl` payload (with its expected SHA, request
IDs, source identities, actor/future schedules, and three-camera locators
cross-checked before any page is staged).  A legacy packet whose manifest has
null request and queue-seal hashes remains compatible, but an unclaimed
request file is never inferred or accepted.  The helper accepts only the
canonical TRAIN handoff; the separate EVAL renderer owned by
`annotate_temporal_questions` produces native packets and an EVAL receipt, not
a TRAIN queue, and must not be restamped or passed through this helper.

`--audit-output` is opt-in and separate.  It contains only
`OFFLINE_FUTURE_AUDIT` samples under `future_audit_pages/`; the default actor
directory and manifest never link to it.  Frame-zero clamping is not
resynthesized: the packet's deduplicated sample list is used as-is and its
`clamped_duplicate_sample_frame_count` is retained in `ordered_events.jsonl`.
No source image is cropped, resized, copied, or edited; only deterministic
black-background composites are written.

Question text is never embedded or rephrased.  With `--query-registry`, exact
event/query IDs and a SHA-256 of the original query text are retained for a
downstream ID join; display names such as `q000` are filenames only and are
not join keys.  The legacy phase40 registry remains on its strict legacy
adapter path.  The versioned
`p107.coverage.visual_relation_query_registry.v1` path is TRAIN-only and
requires the producer's exact `selected_event_bindings.jsonl` receipt.  That
receipt must cover every selected event; an event with no eligible query is
written explicitly as `query_ids: []` with its quarantine/unbound status, and
is never silently treated as a labelable query.  Coverage source pins are
checked against the explicit selector root and byte-pinned provenance bridge
when their original selector/index lineage differs from the canonical RGB
handoff.  EVAL registries and EVAL receipts are rejected before any output.

The builder refuses to overwrite outputs or stale temporary directories and
rejects contradictory role/frame metadata, missing/asymmetric cameras,
duplicate receipt keys, unsafe asset paths, bad hashes, and native image
geometry/mode mismatches.  It cannot add past samples that are absent from the
packet; any longer causal storyboard must first be sealed in the packet's
`causal_temporal_rgb` field.
