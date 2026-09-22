# H09AC: eight fixed development states, two adapters

## Actual diagnostic result,2026-09-22 13:18 BJT

After exact parent releasebc9556316a3dda44fb756655cd0cd20739155de5 /
e46a893a…31de9a, the sole client2839158 started13:15:49.307096 BJT,
immutable19ea, service2833894/9de/identityffea4c97…559c8. Activee9dac7f3…8ed049,
launche016d457…6c980d. Real3.10 complete-input/identity preflight10.035 s passed.
At13:18:33 the client had exited with COMPLETE,16 decisions,22 actual issued
and completed queries,0 errors,120.720811 s. No query retry, training or reset.

| Fixed state | Exact-clock label | Original0120 status | New0120+120 status | New motion if CONTINUE |
| --- | --- | --- | --- | --- |
| FT before00 | CONTINUE | REQUEST_VERIFY | CONTINUE | RIGHT_DOWN |
| FT after01 | CONTINUE | REQUEST_VERIFY | CONTINUE | RIGHT_CLOSE |
| FT after02 | CONTINUE | REQUEST_VERIFY | CONTINUE | RIGHT_UP |
| FT after03 | CONTINUE | REQUEST_VERIFY | CONTINUE | RIGHT_UP |
| FT after04 | CONTINUE | REQUEST_VERIFY | CONTINUE | RIGHT_UP |
| FT after05 | REQUEST_VERIFY | REQUEST_VERIFY | REQUEST_VERIFY | Not queried |
| NN after04 | CONTINUE | REQUEST_VERIFY | CONTINUE | RIGHT_UP |
| NN after05 | REQUEST_VERIFY | REQUEST_VERIFY | REQUEST_VERIFY | Not queried |

Original0120 under the new status interface equals always-request:2 true,
6 early requests,0 true continues. The completion-trained adapter matches all
eight labels. Never-request has6 correct continues and2 missed requests.
An explicitly posthoc public-history rule (at least3 trailing same-hand UP
tokens) also matches all8: this is not visual generalization evidence or a
validated general stopping rule. NN after04 remains the important public
holding=true/strict-completion=false state. Both original physical terminal
failures remain unchanged; no new simulator SR was measured.

Within this new interface, the old adapter never queried motion, so there are
zero jointly-CONTINUE states and conditional null is not an action error.
A separate zero-new-call historical check provides exactly one strict motion
comparison: FT before00 has byte-identical RGB hashes, the exact same six-field
actor and identical1153-token motion input hash1d1f4281…5f5d4 as the old actual
query. Old output RIGHT_BACK changed to new RIGHT_DOWN. The new DOWN was not
executed, so benefit/harm is unknown. Four other historical next-before frames
have matching text-token hashes/motions but different RGB bytes and are not
counted as exact-image comparisons. Motion preservation is not established.

All16 raw responses and observed22-call sequence are local in own artifacts
`h09y-resume-20260921/h09ac_observed_monitor_final.json`, SHA
db10328f99fefd9f08f7e04d8f7c0c0a370b2d8f4b0e085b0cc1be95904a48df.
This is an observed transport copy, not the remote result-file checksum.
At13:24 both the temporary strict stdio transport and original strict SSH
closed during read-only seal checks. The formal remote seal's existence and
complete service-ledger download remain unverified; no blind resubmission was
made. Parent full-result review is pending. No source/result was rewritten.

Implementation closed12:59:07 BJT,1163/1200 s, fixed/pushed
19ea59db69bb33398558d0f3acd8468d9efee53b. Formal remote3.10 runs229 tests/
16.072 s (one optional local-only archive skip); public420/8.969 s is unchanged.
Exact local prepare eab9fbac…362b63 verifies all576 original members and all
eight actors/histories/same-tick labels. Author report6236ab38…3b7ed.
Parent independent229/8.210 s plus21 self-authored checks/4.362 s passed,
review07c3128, file8f25292d…1ec700. No service query is authorized by this
CPU review; actual9de service identity and an independent inference grant
remain prerequisites. No code edit followed the frozen implementation.

Owner Astra-max-vlm_sft_resume_20260921. CPU ticket started2026-09-22
12:39:44 BJT, deadline12:59:44; no new model call, training or simulator reset.
Parent ticketddfb190cc4703e74ab76a601cf41a7405f9bf51c. Only a new independent
evaluator/config/tests are added; active9de training/service, original6f actor,
native45 codec, physical scorers and old failures are not modified.

The fixed inputs are FT i71 before00 and after-settle01–05, plus NN i71
after-settle04–05. All576 original inventory members are byte/SHA checked;
actual completed execution and12 settle controls reconstruct each public
history. The service receives only the original six actor fields and the same
three RGB byte strings for both variants. Paths, clocks, private score, source
variant, expected status and saved verifier results never enter the prompt.

The unchanged exact-clock LocalOutcome yields6 CONTINUE and2 REQUEST_VERIFY;
missing exact endpoints are UNKNOWN. NN after04 has public holding evidence
but remains strict CONTINUE. Always-request has2 true/6 early requests,
including that disagreement; never-request misses2 actual request states.
These are offline labels, not early-stop actions or changed terminal outcomes.
Actual eight-input binding:
971ba21e150b118ae32743e5395726887c331d4a13700cb3541596d2ece60383.

New inference requires exact independent parent/client/service/config/input
and adapter/result bindings. Existing9de service8931 retains64 total calls;
this block is at most16 transactions/32 queries,600 s,16 MiB, no retries.
CONTINUE queries motion on the same snapshot; REQUEST_VERIFY has null motion.
Every failed issued query still belongs to the service ledger. Both original
0120 and0120-plus120 use the identical interface; reference is not a fresh base.
Full raw predictions, motions, early/late confusion and both trivial baselines
are reported. Saved31b/H42 corroboration is labelled zero-new-perception replay.
This is an eight-state development diagnostic, never new simulator SR or
independent generalization evidence. Original FT/NN terminal failures stand.

Local229 native tests/8.134 s pass, including9 new no-privilege/same-pixels,
query-cap/no-retry/deadline/identity/baseline/UNKNOWN cases. Initial two failures
were test-fixture eager clock evaluation and a missing-file exception class;
the production gates were not relaxed. Public regression and formal fixed
prepare are recorded in the handoff. Remote execution and queries await parent
review/release; implementation tests do not self-authorize model loading.
