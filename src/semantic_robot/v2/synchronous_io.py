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

ISAAC = Path('/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/isaacsim')
INSTALLED = {
    ISAAC/'exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py':
        'ebafc6bcb30a454925fe21b96dcdbd4637c922a3fa9d5a6947308c9796ba5028',
    ISAAC/'extscache/omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/omni/replicator/core/scripts/orchestrator.py':
        'c297093ae4100f5fc20f576f0b71d0dc4c765b2135f2a945764bbe63889a7c42',
    ISAAC/'extscache/omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/omni/replicator/core/scripts/annotators.py':
        '4b7b872029493d23e4bbb2a2c6c08ceb353aa1273cba167594421d2eeb3d9379',
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
                 render_time=None):
        if set(references) != VIEWS: raise ValueError('All three camera timestamps required')
        self.sim, self.references, self.capture_native = sim, references, capture
        self.settings, self.record = settings, record
        self.controls = self.captures = self.attempts = 0
        self.release, self.closed = release, False
        self.render_time = render_time  # Diagnostic only; never substitutes the physics clock.
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

    def synchronize(self):
        """Wait once for native annotation completion, without hidden physics."""
        self._check_rates()
        before = self.clock()
        row = {'kind':'capture', 'capture':self.captures+1, 'before':before, 'completed':False}
        old_play = self.settings.get(PLAY_SIMULATIONS)
        if type(old_play) is not bool: raise ValueError('Unknown native playSimulations state')
        try:
            if self.render_time is not None:row['timeline_before']=float(self.render_time())
            self.render_only()
            # Same process-private suppression used by SimulationContext.render.
            # Replicator initialization/updates must never add physical steps.
            self.settings.set(PLAY_SIMULATIONS, False)
            self.capture_native(delta_time=0.0, pause_timeline=False, wait_for_render=True, rt_subframes=4)
            row['after'] = self.clock()
            if self.render_time is not None:row['timeline_after']=float(self.render_time())
            raw = {view:anno.get_data() for view,anno in self.references.items()}
            row['raw_reference_repr'] = {view:repr(value) for view,value in raw.items()}
            seconds = {view:reference_seconds(value) for view,value in raw.items()}
            row['references'] = {view:{k:int(value[k]) for k in ('referenceTimeNumerator','referenceTimeDenominator')}
                                 for view,value in raw.items()}
            row['reference_seconds'] = seconds
            if row['after'] != before: raise RuntimeError('Annotation capture advanced physics')
            if any(abs(t-before['simulation_time'])>1e-6 for t in seconds.values()):
                raise RuntimeError('RGB-D ReferenceTime does not match the current physics state')
            self.captures += 1; row['completed'] = True
            return {view:{'version':'synchronous_io_v1', 'reference_time_seconds':seconds[view],
                          'simulation_time':before['simulation_time'], 'physics_index':before['physics_index'],
                          'wait_for_render':True, 'physics_ticks_in_capture':0,
                          'reference_time_verified':True} for view in VIEWS}
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


def attach_references(sensors, factory, *, edit):
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
                anno=factory('ReferenceTime');references[view]=anno
                anno.attach([paths[view]])
    except BaseException:
        try:release()
        except BaseException as secondary:
            try:print('ReferenceTime partial detach also failed: '+repr(secondary),file=sys.stderr)
            except BaseException:pass
        raise
    return references,release


def configured_adapter(sim, sensors, factory, capture, settings, record, render_time=None):
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
        io=SynchronousIO(sim,references,capture_graph,settings,record,release,render_time)
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
    expected = next(path for path in INSTALLED if path.name == 'orchestrator.py')
    if Path(inspect.getfile(rep.orchestrator.step)).resolve() != expected:
        raise ValueError('Synchronous capture implementation shadowed')
    return configured_adapter(sim,sensors,rep.AnnotatorRegistry.get_annotator,
                              rep.orchestrator.step,carb.settings.get_settings(),record,
                              omni.timeline.get_timeline_interface().get_current_time)
