"""Allow only proved auxiliary renderer contexts from this collection's peers."""
import math
from pathlib import Path


def declared_collection_peers(configured, *, legacy_path=None, legacy_commit=None):
    """Pin all owned collector waves before starting a long-lived RL worker.

    A future wave need not exist yet: declaring it never exempts a process.
    The live PID, exact child directory, frozen commit, other primary GPU and
    <=512MiB auxiliary-only context must still pass owns_collection_auxiliary.
    """
    if not isinstance(configured,list) or bool(legacy_path)!=bool(legacy_commit):
        raise ValueError('Explicit collection peer list and paired legacy binding required')
    rows=list(configured)
    if legacy_path:rows.append(dict(collection=str(legacy_path),source_commit=legacy_commit))
    result=[];seen=set()
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'collection','source_commit'}:
            raise ValueError('Collection peers require exact path and frozen commit only')
        name,commit=row['collection'],row['source_commit']
        if (not isinstance(name,str) or not name or not Path(name).is_absolute()
                or len(Path(name).parts)<4 or any(c in name for c in '*?[]') or '..' in Path(name).parts
                or not isinstance(commit,str) or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit)):
            raise ValueError('No broad, relative, globbed or unpinned peer directory')
        path=str(Path(name).resolve())
        if path in seen:raise ValueError('Duplicate or conflicting collection peer declaration')
        seen.add(path);result.append(dict(collection=path,source_commit=commit))
    return result


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
