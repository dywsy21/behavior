# P107 optional actor-query packer inputs

`pack_memlite_event_labels.py` can add a sealed actor-query sidecar only when
all three inputs are supplied together:

```text
--prelabel-query-registry PRELABEL_REGISTRY.jsonl
--expected-prelabel-query-registry-sha256 <external SHA-256 pin>
--postlabel-view-prelabel-query-bindings VIEW_BINDINGS.jsonl
```

The registry must be the producer's unmodified
`p107.visual_relation_query_prelabel_registry.v1` JSONL. It is validated with
the approved adapter and copied byte-for-byte to `prelabel_queries.jsonl`; its
sealed receipt therefore preserves the original producer SHA, opaque query IDs,
and every source pin. A binding row is exactly:

```json
{"view_id":"<postlabel SHA-256>","prelabel_query_id":"<opaque producer ID>"}
```

Each view and query ID may appear once. The packer rejects an un-packaged view,
an unknown registry ID, or a same-event query whose frozen text/family does not
bind that view's exact goal relation and source skill. It writes only
`actor_queries.jsonl` and `prelabel_queries.jsonl`, registers their byte/row/SHA
receipts in `actor_query_sidecar`, and re-reads them against staged views/events
before the atomic publish.

Omit all three flags for legacy-compatible output. This sidecar does not change
any role, label mask, action payload, dataset-quality gate, or training gate;
in particular it never enables low-action BC or `for_training=True`.
