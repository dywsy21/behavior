"""Allow only proved auxiliary renderer contexts from this collection's peers."""
import math
from pathlib import Path


def owns_auxiliary(pid,used_mib,gpu,receipt,case_directory,argv,environ):
    if (receipt.get('pid')!=pid or not math.isfinite(used_mib) or not 0<=used_mib<=512
            or environ.get('EVAL_GPU') not in {str(i) for i in range(8) if i!=gpu}):return False
    if '--output' not in argv:return False
    index=argv.index('--output')
    if index+1>=len(argv) or Path(argv[index+1]).resolve()!=Path(case_directory).resolve():return False
    names={Path(v).name for v in argv}
    return bool(names & {'collect_local_grasp_recovery.py','collect_local_articulation_recovery.py',
                         'collect_placement_curriculum.py'})
