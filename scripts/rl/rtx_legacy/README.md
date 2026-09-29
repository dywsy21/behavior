# Pinned simulator-side legacy dependencies

These four files are byte-preserved copies of the team's existing robo runtime,
vendored so the independent RTX probe can be deployed through Git. The original
live files are untouched. This is not a new model, controller or oracle policy.
The package initializer is deliberately minimal; serving-only re-exports are
not needed by the simulator factory.

| File | SHA256 |
| --- | --- |
| native_oracle_low_v1/official_factory.py | 5492910dac0a75aac9e8ca4e71f5b762aeff4e129c37d6ed4d8c995c312e5f70 |
| native_oracle_low_v1/runtime.py | 61beb65dbe0dc5a6e7bb952a60a9d790c9b2eb51bd546e2ecc53a61fb662c447 |
| run_behavior_eval_chunked.py | 8feddedc46ffca3cb63bd80b4c237e9a157a3c8c7929c74b74c5dbc51ec524c5 |
| memlite_sim_trace.py | d2f9eb7df66c982aac0a15c34a8768a75215370bb72d178b0ce596bc72b0bef5 |

Factory/runtime origin:
`/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/production_native_oracle_low_v1/native_oracle_low_v1/`.
Chunk/trace origin: `/mnt/sdc1/robodojo/behavior_eval/`.
Runtime path overrides must be explicit and preserve these import hashes. No
weights, datasets, license keys, runtime caches or credentials belong here.
