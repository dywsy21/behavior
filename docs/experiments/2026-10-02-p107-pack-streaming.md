# P107 sealed package streaming scope

`pack_memlite_event_labels.py` now reads the small annotation document first,
then streams the immutable event JSONL while retaining only the event IDs bound
by those views.  The full source-group inventory and every event row still pass
strict JSON, duplicate-ID, source-group, and episode-lineage validation; both
sealed payload files must match their manifest SHA-256 and byte receipts before
the package is trusted.  The emitted `events.jsonl` remains the complete,
canonical source index and is copied in a second sealed streaming pass.

This is a bounded-memory publisher implementation change only.  It does not
alter the P107 protocol/index schema, annotation semantics, training gates, or
release authorization; `ready_for_training` remains fail-closed.
