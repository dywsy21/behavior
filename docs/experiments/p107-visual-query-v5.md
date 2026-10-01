# P107 phase40 visual-query producer v5

v5 is a narrow correction over v4.  The pinned official BDDL category
vocabulary is unchanged (`bd049de3119acdcdf2334fe9e1ebe060fa20c108`,
`category_mapping.csv`, SHA-256
`ef4636716bc1f243f89735ffe89e8e931a2d5739e87164e7146d54d11c681eab`).

The resolver now accepts both forms:

1. an exact raw token equal to an official category, such as
   `electric_switch` or `bottom_cabinet`; and
2. a raw token beginning with an official category plus `_`, selecting the
   unique longest boundary-aware prefix, such as
   `half_log_176_0` → `half log`.

For exact category tokens, `opaque_instance_suffix` is `null`.  For suffixed
tokens, the suffix remains opaque and is retained only in the category audit
sidecar.  No-match and tied-prefix inputs remain `UNKNOWN_CATEGORY` and are
quarantined; no aliases, fuzzy matching, or fixed-width suffix stripping are
used.  The v4 output remains frozen; v5 is a new external artifact.

The public prelabel registry schema, role ambiguity requirements, RGB/state
gates, future-frame policy, calibration-only status, and agent provenance are
unchanged.  The producer still emits no answers, labels, action payloads,
outcome claims, or postlabel view IDs.

The sealed v5 output is built from the same phase40 inputs with the pinned CSV
as `--category-mapping`, into the new external directory
`/home/wsy/behavior-annotations/p107/phase-balanced-calibration40-v5-questions`.
