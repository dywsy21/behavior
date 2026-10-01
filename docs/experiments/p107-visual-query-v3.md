# P107 phase40 prelabel visual-query producer v3

This revision withdraws the v2 query text for metadata-quality reasons; it
does not alter the sealed phase queue or create labels.  The corrected output
is a new external directory, `phase-balanced-calibration40-v3-questions`.

## v3 wording contract

- Entity display names remove only an explicit final numeric instance suffix.
  Nonnumeric noun tokens are retained.  If multiple IDs collapse to one
  display name, the question preserves multiplicity rather than guessing an
  identity.
- `OPEN/CLOSE_DOOR`, `OPEN/CLOSE_DRAWER`, and `OPEN/CLOSE_LID` name the
  controlled part and its parent.  A metadata `target_part` is retained; when
  absent, the canonical skill supplies only the controlled-part noun (for
  example, a toolbox lid or cabinet drawer), without claiming a hidden
  mechanism.
- Effect queries name the metadata-derived target/reference entities in the
  text.  Ambiguous source/material/surface roles remain explicitly subject to
  visual grounding; the producer does not invent a destination or recipient.
- Placement queries with no explicit destination list the named reference
  entities and retain the unresolved target/reference-role gate.
- The current query remains one atomic visible relation at the anchor.  Any
  historical change relation is separate and requires an observable prestate
  and poststate.  `PRESS`, `HANDOVER`, and `NAVIGATE` retain their v2 safety
  gates: body contact is not a press result, recipients are not assumed to be
  people, and metric distance/pose requires state evidence.

## Identity and routing

The prelabel query ID remains answer-independent and is recomputed from the
corrected query identity plus sealed source pins.  v3 records remain
calibration-only (`training_eligible=false`), contain no answers/RGB evidence,
future actor references, outcome, recovery, or action payload, and retain
`gpt-5.6-luna/max`, `human_reviewed=false`, and `image_inspected=false`
provenance.  The v2 external output is immutable and withdrawn pending review;
consumers must use the v3 manifest and its hashes.
