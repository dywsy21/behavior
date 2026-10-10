"""Fresh-process worker boundaries; a failed reset is NEVER a retry signal."""


def require_service_hello(message, config_sha256):
    """A transport preflight does not claim a job or acknowledge completion."""
    if message != dict(protocol='short_skill_rl_v1', config_sha256=config_sha256):
        raise ValueError('Service protocol/config differs; do not load a simulator or retry this binding')
    return True


def probe_finish_without_simulator(completed, seeds):
    """Only after every fixed read-only seed has a closed simulator receipt."""
    if (type(completed) is not int or completed<0 or not seeds
            or any(type(seed) is not int for seed in seeds) or len(set(seeds))!=len(seeds)
            or completed>len(seeds)):
        raise ValueError('Exact nonduplicate read-only episode accounting required')
    return completed==len(seeds)


def require_probe_finish_response(response):
    """A finished probe case may wait for peers, but must never take a new job."""
    if response=={'status':'finished'}:return True
    if response=={'status':'wait'}:return False
    raise ValueError('Unexpected job/error after all fixed probe episodes; do not run/retry it')


def validate_baseline_acceptance(receipt, *, config_sha256, policy_sha256, episodes_sha256):
    expected = dict(schema='short_skill_cold_baseline_acceptance_v1', approved=True,
        config_sha256=config_sha256, policy_sha256=policy_sha256,
        episodes_sha256=episodes_sha256, reviewed_episodes=6)
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Baseline acceptance must bind all six real cold starts and this exact resumed policy')


def finished_workers(cases, acknowledged, case):
    """Do not close the service before every cold worker sees explicit finish."""
    if case not in cases:
        raise ValueError('Unregistered worker cannot acknowledge completion')
    acknowledged.add(case)
    return acknowledged == set(cases)


def next_worker_action(returncode, receipt):
    if returncode != 0 or not isinstance(receipt,dict):
        raise ValueError('Simulator process failed or omitted its terminal receipt')
    status = receipt.get('status')
    if status == 'completed_single_episode':
        if receipt.get('completed_episodes') != 1 or receipt.get('fresh_process_per_episode') is not True:
            raise ValueError('Invalid fresh-process completion boundary')
        return 'next_episode'
    if status == 'server_finished' and receipt.get('completed_episodes') == 0:
        return 'finished'
    # Even a graceful websocket close can be another worker's failure. Only
    # the service's explicit finished message means the window completed.
    raise ValueError('Do not retry a failed/unknown start or infer training completion from disconnect')
