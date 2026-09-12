# v9 5000-step sampling allocation audit

This replays the actual source `ResumableDistributedBranchBalancedBatchSampler` with the saved seed7, batch8, four ranks, one high + seven low per rank, epoch0 and first5000 batches. It reads the approved sidecar and actual episode metadata, reproduces the source task-stratified split and compact high/low pools. It does not load the model, images or simulator, and does not modify any running evaluation.

These are **scheduled sample indices**, not a row-by-row receipt of examples finally delivered after dataset resampling. They quantify exposure, not a causal explanation or a success metric.

| Task | Eligible low pool | Scheduled low samples | Low share | Scheduled high samples |
|---|---:|---:|---:|---:|
| Radio | 377,625 | 6,138 | 4.38% | 972 |
| Trash | 883,099 | 14,359 | 10.26% | 2,155 |
| Halloween | 2,591,068 | 42,113 | 30.08% | 5,984 |
| Plates/food | 2,555,800 | 41,542 | 29.67% | 5,817 |
| Can meat | 2,205,282 | 35,848 | 25.61% | 5,072 |
| Total | 8,612,874 | 140,000 | 100% | 20,000 |

All140,000 scheduled low indices and all20,000 high indices are unique in this prefix. Low exposure is about1.625% of its eligible frame pool. This percentage alone does not establish underfitting: adjacent frames are correlated and pretrained knowledge matters.

The sampling design balances branches, not tasks. Long demonstrations therefore contribute more examples even though the current diagnostic success summary gives each task one rollout. Task-balanced or mixed task/frame sampling is a justified next experiment, not an already demonstrated improvement. No new training or sampler changes have been launched from this audit.

Evidence: `memlite-v9-training-allocation.json` and `audit_memlite_v9_training_allocation.py`. Remote artifacts: `robo:/mnt/sdc1/robodojo/behavior_dev/memlite_v9.KWefx2/training_allocation_replay_20260907.json` and the corresponding audit script. Metadata selection was asserted to950 original training and50 eval episodes, with training Dataset length8,898,502; sidecar exclusions and high-priority duplicate-frame removal give the smaller eligible pools above.
