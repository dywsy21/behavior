# H09AB: separately queried completion requests

Owner: Astra-max-vlm_sft_resume_20260921. CPU implementation started
2026-09-22 11:39:40 BJT; deadline 12:19:40 BJT (2400 s). Parent ticket
47a85290d2d46f96e29db3551dcc1bb74e08eee6. This ticket permits no weight
loading, neural queries, optimizer updates, simulation or resets.

Hypothesis: retaining motion imitation while learning a separate completion
request can avoid unnecessary continued motion after a locally completed skill.
This is not yet a demonstrated effect; original i71 FT and NN terminal failures
remain failures despite their independently verified transient local successes.

## Fixed inputs and output separation

H09AA ccb16f2 supplies exactly 39 reviewed states (34 CONTINUE, 5 REQUEST_VERIFY),
states SHA 2fd27d707ad86e7fd6c09662000e99af26c0b1d686bbe9ff92b3c6aa8604fa99,
manifest SHA 085d0a06eb590d0f390337951ca47ba412009010e69cd51a8665d838e8aba0fd.
Parent data review403a4a0 SHA7fbb349f81a71d944696cad4ad0d58b4766e4bde9d1b7a014714bbee9f7bbd2f
binds the original five physical audits and all30 terminal/last-before RAWs.
Original34 motion rows, successful-run/coverage gates and all old records stay
unchanged. There is no terminal HOLD motion target and no heldout label input.

New modules only: completion_protocol/modeling/training_data/runtime/train/
serve/preflight. Public executor remains8fdfcd3e1b8bdfd95530eec09710d1c44efc215885b49c64c1a82f55bbc501a8,
core0808ee8d811b78d1cd34d1da33c30abc705a66d1; old native_train, native_eval,
native_serve, actor projection, native45 codec and public source are unmodified.

The status formatter uses only the same six public actor fields and current
three RGB images. It asks only CONTINUE/REQUEST_VERIFY, not the old Next motion
prompt. The motion formatter reproduces the old message/image prefix exactly.
An internal status→motion service transaction freezes the actor and pixels;
REQUEST_VERIFY returns null motion. Both adapters use this identical interface.
It cannot execute, certify holding, or claim skill/official success.

## Future single training run — not authorized by this document

Warmstart only original training_v1/adapter_0120, adapter SHA
4e993ff4f5ca7221e4fcdc61ad302ac644b3a27fe7ef2c25b57516ac21debf83.
Actual serialized PEFT config contains module suffixes, not full paths; its
SHA62621ecba00f55f36658d3161c3e7ba38a977b3f5362ff51abee96fbc32e8a21
is pinned and the loaded trainable set must remain372 language LoRA tensors /
16,819,200 parameters. No vision/projector parameters become trainable.

One new120-update run, first2 actual gradient/save-reload gate updates included;
2700 s, GPU3, seed41, r16/alpha32/dropout.05, LR5e-5, batch8/micro2. Each update
uses two motion microbatches, one CONTINUE microbatch and one REQUEST_VERIFY
microbatch, each weighted1/4; trajectory-balanced sampling within each category.
No heldout checkpoint selection. All73 query response/EOS masks and each
category's native/custom CE are gated; update2 reload checks all three categories.

Parent confirmed new ROOT/training_completion_v1 and ROOT/service_completion_v1,
each384 MiB under existing6 GiB result root. Model-only cache directories under
the existing16 GiB runtime root; no new OG alias, old cache moves or HOME change.
NVMe≥80 GiB/SDA≥32 GiB remain mandatory, with source/config/data/parent-release
checks and per-update/query storage checks. New port8931 must be unoccupied.
Service variants motion_only_0120 and completion_0120_plus120 share64 TOTAL
calls, counting each status and motion separately, including failed issued
queries. No retry or repeated request id. Old2762269/8919 is untouched.

## Validation state and remaining bridge

Author local220 native tests/6.483 s pass, including12 new tests; first public
test invocation omitted PYTHONPATH and had20 import errors, not production
failures. Correct source-path public420/7.370 s passes. Fixed351b279 remote
3.10/NumPy1.26/transformers5.7/PEFT0.18 passes220 tests/15.303 s (one optional
local-archive test skipped). Real CPU preflight22.891 s passes all73 response
masks/train-infer prefixes and all34 exact old encodings, no model weights or
CUDA. Its first report inadvertently let the data's source_commit overwrite
the preflight source_commit in the JSON merge; preserved as v1 and corrected
by naming provenance data_source_commit. No model input/target or source gate
changed. A new immutable source/v2 report is required for final handoff.

Parent H43 owns verify_grasp_request and physical capture/history validation.
This ticket only exposes a callback bridge; a future jointly reviewed runtime
must bind its current capture to the status-query snapshot and recheck before
any execution. Public holding is not skill/official success. No simulator
integration, evaluation reset, service launch or optimizer update is granted
by these implementation files. Parent independent review is still required.
