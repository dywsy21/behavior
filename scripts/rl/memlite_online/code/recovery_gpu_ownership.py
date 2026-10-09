"""Allow only proved auxiliary renderer contexts from this collection's peers."""
import math
from pathlib import Path


def owns_auxiliary(pid,used_mib,gpu,receipt,case_directory,argv,environ,*,collectors=None):
    if (receipt.get('pid')!=pid or not math.isfinite(used_mib) or not 0<=used_mib<=512
            or environ.get('EVAL_GPU') not in {str(i) for i in range(8) if i!=gpu}):return False
    if '--output' not in argv:return False
    index=argv.index('--output')
    if index+1>=len(argv) or Path(argv[index+1]).resolve()!=Path(case_directory).resolve():return False
    names={Path(v).name for v in argv}
    allowed=({'collect_local_grasp_recovery.py','collect_local_articulation_recovery.py',
              'collect_placement_curriculum.py'} if collectors is None else collectors)
    return bool(names & allowed)


def owns_short_skill_auxiliary(pid,used_mib,gpu,receipt,case_directory,argv,environ,*,config_sha256,cases,port):
    if (receipt.get('config_sha256')!=config_sha256 or receipt.get('case') not in cases
            or '--port' not in argv or argv[argv.index('--port')+1:] == []
            or argv[argv.index('--port')+1]!=str(port)):
        return False
    return owns_auxiliary(pid,used_mib,gpu,receipt,case_directory,argv,environ,
                          collectors={'collect_short_skill_rl.py'})
