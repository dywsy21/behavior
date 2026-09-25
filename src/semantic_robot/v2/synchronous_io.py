"""Opt-in clocked control and timestamped RGB-D. No scene/pose truth.

The native adapter only attaches time annotators to existing render products.
Physics advancement and rendering are deliberately separate transactions.
"""
from contextlib import contextmanager
import hashlib
import inspect
import math
from pathlib import Path
import sys
from .render_batch import (FRAME_ANNOTATOR, validate_batch, validate_advance,
                           make_receipts, native_batch, register_frame_annotator)

ISAAC = Path('/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/isaacsim')
INSTALLED = {
    ISAAC/'exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py':
        'ebafc6bcb30a454925fe21b96dcdbd4637c922a3fa9d5a6947308c9796ba5028',
    ISAAC/'extscache/omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/omni/replicator/core/scripts/orchestrator.py':
        'c297093ae4100f5fc20f576f0b71d0dc4c765b2135f2a945764bbe63889a7c42',
    ISAAC/'extscache/omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/omni/replicator/core/scripts/annotators.py':
        '4b7b872029493d23e4bbb2a2c6c08ceb353aa1273cba167594421d2eeb3d9379',
    ISAAC/'extscache/omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/omni/replicator/core/scripts/annotators_default.py':
        'a9a8e105828c14731fe00f38d01c872611cec8364d4287c8da2bd73f1ca9dcba',
    ISAAC/'extscache/omni.syntheticdata-0.6.13+69cbf6ad.lx64.r.cp311/omni/syntheticdata/scripts/SyntheticData.py':
        '18cac198726aee05c2b5f56cd49b28879217bc37fbbb55ede090ac8d41060c93',
    Path('/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson/omnigibson/sensors/vision_sensor.py'):
        'a445dc5d73ea2d02fa0a87b1cf688abe52b1f213d9263b3e2cfd1fc1e46b9e79',
}
VIEWS = {'head', 'left_wrist', 'right_wrist'}
PLAY_SIMULATIONS = '/app/player/playSimulations'


def native_clock(sim):
    value = {'simulation_time': float(sim.current_time),
             'physics_index': int(sim.current_time_step_index)}
    if not math.isfinite(value['simulation_time']) or value['physics_index'] < 0:
        raise ValueError('Invalid native simulation clock')
    return value


@contextmanager
def graph_edit(edit):
    """Use OG's USD/Fabric transaction without hiding the original error.

    The installed context is non-nestable and synchronizes Fabric on exit.
    A synchronization error must not replace an earlier graph/API error.
    """
    context = edit()
    context.__enter__()
    try:
        yield
    except BaseException:
        primary = sys.exc_info()
        try: context.__exit__(*primary)
        except BaseException as secondary:
            try: print('USD/Fabric context exit also failed: '+repr(secondary),file=sys.stderr)
            except BaseException: pass
        raise
    else:
        context.__exit__(None,None,None)


def validate_installed():
    for path, expected in INSTALLED.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Synchronous I/O native dependency changed: '+str(path))


def reference_seconds(value):
    # Reject missing/zero/float timestamps, rather than silently manufacturing
    # a "current" timestamp for a not-yet-initialized sensor.
    if not isinstance(value, dict): raise ValueError('Missing native ReferenceTime')
    numerator, denominator = (value.get('referenceTime'+name) for name in ('Numerator', 'Denominator'))
    # Native integer scalars may be numpy.uint64, not Python int.
    import numbers
    if (any(isinstance(x, bool) or not isinstance(x, numbers.Integral) for x in (numerator, denominator))
            or numerator < 0 or denominator <= 0):
        raise ValueError('Invalid native ReferenceTime fraction')
    return int(numerator)/int(denominator)


class SynchronousIO:
    def __init__(self, sim, references, capture, settings, record, release=lambda:None,
                 render_time=None, batch_state=None):
        if set(references) != VIEWS: raise ValueError('All three camera timestamps required')
        self.sim, self.references, self.capture_native = sim, references, capture
        self.settings, self.record = settings, record
        self.controls = self.captures = self.attempts = 0
        self.release, self.closed = release, False
        self.render_time = render_time  # Diagnostic only; never substitutes the physics clock.
        self.batch_state = batch_state
        self.primes = self.reads = 0
        self.last_batch = self.pending = None
        self._check_rates()

    def _check_rates(self):
        if getattr(self,'closed',False): raise RuntimeError('Synchronous I/O already closed')
        rates = (float(self.sim.get_physics_dt()), float(self.sim.get_sim_step_dt()),
                 float(self.sim.get_rendering_dt()))
        if any(not math.isfinite(a) or abs(a-b)>1e-10 for a,b in zip(rates,(1/120,1/30,1/30))):
            raise ValueError('Registered 120Hz physics/30Hz action/render rates changed')

    def clock(self):
        return native_clock(self.sim)

    def _record(self, row):
        if sys.exc_info()[0] is None:
            self.record(row)
        else:
            try: self.record(row)
            except BaseException as secondary:
                try: print('Synchronous I/O evidence save also failed: '+repr(secondary),file=sys.stderr)
                except BaseException: pass

    def _failure(self, row, error):
        row['error'] = repr(error)
        try: row['after'] = self.clock()
        except BaseException as secondary: row['after_clock_error'] = repr(secondary)

    @contextmanager
    def control(self, *, render_requested):
        self._check_rates()
        if self.pending is not None: raise RuntimeError('Unverified RGB-D read before control')
        before = self.clock()
        self.attempts += 1
        row = {'kind':'control', 'call':self.attempts, 'before':before, 'completed':False,
               'render_requested':bool(render_requested), 'actual_render_on_step':False}
        try:
            with self.sim.render_on_step(False):
                yield
            after = self.clock(); row['after'] = after
            if (after['physics_index']-before['physics_index'] != 4 or
                    abs(after['simulation_time']-before['simulation_time']-1/30)>1e-7):
                raise RuntimeError('Control did not advance exactly four physics ticks / 1/30s')
            self.controls += 1; row['completed'] = True
        except BaseException as error:
            self._failure(row,error)
            raise
        finally:
            self._record(row)

    def close(self):
        if self.closed:return
        self.closed=True  # Never retry a failed native detach indefinitely.
        row={'kind':'close','completed':False}
        primary=sys.exc_info()[1]
        try:
            # Still release graph references if reading the clock itself fails.
            try: row['before']=self.clock()
            except BaseException as error: row['before_clock_error']=repr(error)
            self.release()
            row['after']=self.clock()
            if row.get('before')!=row['after']:raise RuntimeError('I/O close clock changed or unavailable')
            row['completed']=True
        except BaseException as error:
            self._failure(row,error)
            if primary is None:raise
        finally:
            self._record(row)

    def render_only(self):
        before = self.clock()
        self.sim.render()  # Official Fabric update plus render-only app update.
        if self.clock() != before: raise RuntimeError('Render-only call advanced physics')

    def _capture_batch(self, row):
        self.capture_native(delta_time=0.0, pause_timeline=False, wait_for_render=True, rt_subframes=4)
        row['after'] = self.clock()
        if row['after'] != row['before']: raise RuntimeError('Annotation capture advanced physics')
        row['batch'] = self.batch_state()
        validate_batch(row['batch'])
        return row['batch']

    def _prime(self):
        row = {'kind':'prime', 'before':self.clock(), 'completed':False, 'discarded':True}
        try:
            self.render_only()
            # SimulationContext.render restores this setting; suppress again.
            self.settings.set(PLAY_SIMULATIONS, False)
            self._capture_batch(row)
            self.primes = 1
            row['completed'] = True
        except BaseException as error:
            self._failure(row,error)
            raise
        finally:
            self._record(row)

    def synchronize(self):
        """Capture a fresh render batch; the caller must subsequently verify_read."""
        self._check_rates()
        if self.pending is not None: raise RuntimeError('Previous RGB-D read not verified')
        if not callable(self.batch_state): raise ValueError('Native per-product batch reader required')
        before = self.clock()
        row = {'kind':'capture', 'capture':self.captures+1, 'before':before, 'completed':False}
        old_play = self.settings.get(PLAY_SIMULATIONS)
        if type(old_play) is not bool: raise ValueError('Unknown native playSimulations state')
        try:
            if self.render_time is not None:row['timeline_before']=float(self.render_time())
            self.settings.set(PLAY_SIMULATIONS, False)
            if not self.primes: self._prime()
            self.render_only()
            # Same process-private suppression used by SimulationContext.render.
            # Replicator initialization/updates must never add physical steps.
            self.settings.set(PLAY_SIMULATIONS, False)
            row['baseline'] = self.batch_state()
            validate_batch(row['baseline'], matched=False)
            batch = self._capture_batch(row)
            if self.render_time is not None:row['timeline_after']=float(self.render_time())
            validate_advance(batch, row['baseline'])
            if self.last_batch is not None: validate_advance(batch, self.last_batch)
            self.captures += 1; row['completed'] = True
            self.pending = {'clock':before, 'batch':batch, 'capture':self.captures}
            return make_receipts(batch,before,self.captures)
        except BaseException as error:
            self._failure(row,error)
            raise
        finally:
            primary = sys.exc_info()[1]
            try: self.settings.set(PLAY_SIMULATIONS, old_play)
            except BaseException as secondary:
                row.update(restore_error=repr(secondary),completed=False)
                if primary is None:
                    self._record(row)
                    raise
            self._record(row)

    def verify_read(self, receipts):
        """Fence AFTER all RGB/depth buffers have been copied, before any use."""
        self._check_rates()
        if self.pending is None: raise RuntimeError('No pending RGB-D capture')
        pending = self.pending
        row = {'kind':'read', 'capture':pending['capture'], 'before':self.clock(), 'completed':False}
        try:
            row['batch'] = self.batch_state()
            row['after'] = self.clock()
            if row['before'] != pending['clock'] or row['after'] != pending['clock']:
                raise RuntimeError('Physics changed while copying RGB-D')
            if row['batch'] != pending['batch']:
                raise RuntimeError('Render batch changed while copying RGB-D')
            expected = make_receipts(pending['batch'],pending['clock'],pending['capture'])
            if set(receipts) != VIEWS or any(receipts[v].get('native_time') != expected[v] for v in VIEWS):
                raise ValueError('RGB-D read receipt does not bind the pending batch')
            row['buffer_hashes'] = {v:{k:receipts[v][k] for k in ('rgb_sha256','depth_sha256')} for v in VIEWS}
            if any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h)
                   for hashes in row['buffer_hashes'].values() for h in hashes.values()):
                raise ValueError('Copied RGB-D buffer hashes required')
            self.last_batch = pending['batch']; self.pending = None
            self.reads += 1; row['completed'] = True
        except BaseException as error:
            self._failure(row,error)
            raise
        finally:
            self._record(row)

    def abandon_read_after_failure(self):
        """Only exception cleanup may discard a pending read for a safe hold."""
        if sys.exc_info()[0] is None: raise RuntimeError('Cannot discard a pending read on the normal path')
        if self.pending is None:return
        row={'kind':'abandoned_read','capture':self.pending['capture'],
             'completed':False,'error':repr(sys.exc_info()[1])}
        self.pending=None
        try:row['before']=row['after']=self.clock()
        except BaseException as secondary:row['clock_error']=repr(secondary)
        self._record(row)


def attach_references(sensors, factory, *, edit, annotator_name=FRAME_ANNOTATOR):
    """Exception-safe attachment to existing products; never recreate them."""
    references,paths={},{}
    # Snapshot real paths before creating any graph node. Replicator 1.12.27
    # accepts HydraTexture on attach, but its detach(list) fails to unwrap it.
    for view,sensor in sensors.items():
        product=sensor.render_product
        path=product if isinstance(product,str) else getattr(product,'path',None)
        if not isinstance(path,str) or not path.startswith('/'):
            raise ValueError('Existing render product path required')
        paths[view]=path
    def release():
        with graph_edit(edit):
            errors=[]
            for view,anno in references.items():
                try:anno.detach([paths[view]])
                except BaseException as error:errors.append(error)
            if errors:raise errors[0]
    try:
        with graph_edit(edit):
            for view in sensors:
                anno=factory(annotator_name);references[view]=anno
                anno.attach([paths[view]])
    except BaseException:
        try:release()
        except BaseException as secondary:
            try:print('Frame marker partial detach also failed: '+repr(secondary),file=sys.stderr)
            except BaseException:pass
        raise
    return references,release


def configured_adapter(sim, sensors, factory, capture, settings, record, render_time=None, batch_reader=None):
    """Clock-audited lifecycle, also executable without the native SDK in tests."""
    if set(sensors) != VIEWS: raise ValueError('Exact existing onboard cameras required')
    row={'kind':'initialize','before':native_clock(sim),'completed':False}
    release=None
    try:
        references,release=attach_references(sensors,factory,edit=sim.editing_usd)
        def capture_graph(**kwargs):
            # Orchestrator initialization also creates graph nodes. Do not
            # disable OG's guard or wrap an entire action/episode in this scope.
            with graph_edit(sim.editing_usd):capture(**kwargs)
        io=SynchronousIO(sim,references,capture_graph,settings,record,release,render_time,
                         (lambda:batch_reader(references)) if batch_reader is not None else None)
        row['after']=io.clock()
        if row['after']!=row['before']:raise RuntimeError('I/O initialization advanced physics')
        row['completed']=True
        record(row)
        return io
    except BaseException as error:
        row.update(error=repr(error),completed=False)
        if release is not None:
            try:release()
            except BaseException as secondary:row['cleanup_error']=repr(secondary)
        try:row['after']=native_clock(sim)
        except BaseException as secondary:row['after_clock_error']=repr(secondary)
        try:record(row)
        except BaseException as secondary:
            try:print('I/O initialization evidence also failed: '+repr(secondary),file=sys.stderr)
            except BaseException:pass
        raise


def native_adapter(sim, sensors, record):
    validate_installed()
    import carb.settings
    import omni.replicator.core as rep
    import omni.timeline
    import omni.graph.core as graph
    from omni.syntheticdata import SyntheticData
    expected = next(path for path in INSTALLED if path.name == 'orchestrator.py')
    if Path(inspect.getfile(rep.orchestrator.step)).resolve() != expected:
        raise ValueError('Synchronous capture implementation shadowed')
    register_frame_annotator(rep,SyntheticData)
    return configured_adapter(sim,sensors,rep.AnnotatorRegistry.get_annotator,
                              rep.orchestrator.step,carb.settings.get_settings(),record,
                              omni.timeline.get_timeline_interface().get_current_time,
                              lambda references:native_batch(sensors,references,rep.orchestrator,graph))
