"""Process-local launch adapter; never edits installed OmniGibson/Isaac files.

Keep the installed OG launch sequence (assets/MDL/stage/backend/hooks), changing
only its two redundant apps copies to content-verified no-ops and delegating
SimulationApp construction to the separately registered bounded profile.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from threading import Lock


_GATE = Lock()
_ACTIVE_PATHS = set()
_CONSUMED_PATHS = set()


class _Proxy:
    def __init__(self, original, **overrides):
        self._original, self._overrides = original, overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._original, name)


@contextmanager
def private_og_startup(module, *, source_sha256, experience, copy_bindings, construct):
    """Exactly one OG launch in this process; restore module references on exit.

    copy_bindings maps (source, existing destination) to their common SHA256.
    Unexpected copies, changed installed bytes or an already live app fail.
    No global shutil or installed isaacsim module is monkey-patched.
    """
    if not callable(construct) or module.og.app is not None:
        raise ValueError('Fresh process and explicit registered constructor required')
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != source_sha256:
        raise ValueError('Installed OG startup source changed')
    if len(copy_bindings) != 2:
        raise ValueError('Exact existing OG experience and icon copies required')
    bindings = {(Path(src).resolve(), Path(dst).resolve()): sha
                for (src, dst), sha in copy_bindings.items()}
    if len(bindings) != 2:
        raise ValueError('Copy targets alias each other')
    experience = Path(experience).resolve()
    if experience not in {dst for _, dst in bindings}:
        raise ValueError('Experience is not one of the frozen copy targets')
    for pair, sha in bindings.items():
        if any(not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha for path in pair):
            raise ValueError('Existing OG apps resources are not byte-identical')
    source_path = Path(module.__file__).resolve()
    with _GATE:
        if source_path in _ACTIVE_PATHS or source_path in _CONSUMED_PATHS:
            raise RuntimeError('OG startup already reserved or attempted in this process')
        _ACTIVE_PATHS.add(source_path)
    active = True
    original_launch, original_lazy, original_shutil = module._launch_app, module.lazy, module.shutil
    receipt = {'launch_calls': 0, 'app_constructions': 0, 'verified_copy_noops': [],
               'shared_install_writes': 0, 'original_launch_source_sha256': source_sha256}

    def verified_noop(source, destination, *args, **kwargs):
        pair = (Path(source).resolve(), Path(destination).resolve())
        if args or kwargs or pair not in bindings or any(
                [str(p) for p in pair] == row['paths'] for row in receipt['verified_copy_noops']):
            raise ValueError('Unregistered or repeated shared-app copy requested')
        digest = bindings[pair]
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path in pair):
            raise ValueError('Shared apps bytes changed before launch')
        receipt['verified_copy_noops'].append({'paths': [str(p) for p in pair], 'sha256': digest})
        return str(destination)

    def app_factory(config, *, experience):
        expected = {'headless': True, 'multi_gpu': False, 'active_gpu': 3, 'physics_gpu': 3}
        if (set(config) != set(expected) or any(type(config[k]) is not type(v) or config[k] != v
                                               for k, v in expected.items())):
            raise ValueError('OG requested a different headless/GPU configuration')
        if Path(experience).resolve() != frozen_experience or receipt['app_constructions']:
            raise ValueError('Changed experience or repeated Kit constructor')
        if len(receipt['verified_copy_noops']) != 2:
            raise ValueError('App construction preceded frozen-resource verification')
        receipt['app_constructions'] += 1
        return construct()

    frozen_experience = experience
    def launch_once():
        with _GATE:
            if not active or source_path in _CONSUMED_PATHS:
                raise RuntimeError('This profile permits only one active OG launch attempt')
            # Consume before any import/construction. A failed attempt may have
            # initialized native state, so it cannot be retried in this process.
            _CONSUMED_PATHS.add(source_path)
            receipt['launch_calls'] += 1
        # These proxies exist only on the OG simulator module and only during
        # its original synchronous launch; all other attributes are unchanged.
        try:
            module.shutil = _Proxy(original_shutil, copyfile=verified_noop)
            module.lazy = _Proxy(original_lazy,
                isaacsim=_Proxy(original_lazy.isaacsim, SimulationApp=app_factory))
            result = original_launch()
            if receipt['app_constructions'] != 1 or len(receipt['verified_copy_noops']) != 2:
                raise RuntimeError('Original OG launch did not follow the reviewed route')
            return result
        finally:
            module.shutil, module.lazy = original_shutil, original_lazy

    module._launch_app = launch_once
    try:
        yield receipt
    finally:
        with _GATE:
            active = False
            _ACTIVE_PATHS.remove(source_path)
        module._launch_app = original_launch
        module.shutil, module.lazy = original_shutil, original_lazy
