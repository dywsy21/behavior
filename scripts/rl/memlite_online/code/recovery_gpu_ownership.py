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


def owns_collection_auxiliary(pid,used_mib,gpu,receipt,case_directory,argv,environ,*,collection,source_commit):
    """An explicit frozen collection, not any small process, may be a peer."""
    if (receipt.get('schema')!='local_recovery_collection_status_v1'
            or receipt.get('source_commit')!=source_commit
            or len(source_commit)!=40 or any(c not in '0123456789abcdef' for c in source_commit)
            or Path(case_directory).resolve().parent!=Path(collection).resolve()):
        return False
    return owns_auxiliary(pid,used_mib,gpu,receipt,case_directory,argv,environ)
