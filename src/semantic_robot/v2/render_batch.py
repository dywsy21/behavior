"""Render-domain evidence, deliberately separate from the physics clock.

No object pose, segmentation or privileged task state is queried here.
Per-product frame IDs plus the pinned synchronous API fence RGB-D reads;
we do not pretend that three copies of global ReferenceTime do so.
"""
from fractions import Fraction
import numbers

VIEWS = {'head', 'left_wrist', 'right_wrist'}
FRAME_ANNOTATOR = 'BehaviorRenderBatchV1'
FRAME_FIELDS = ('type', 'frameNumber', 'rationalTimeOfSimNumerator',
                'rationalTimeOfSimDenominator')


def integer(value, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError('Invalid native render integer: '+repr(value))
    return int(value)


def fraction(value):
    if not isinstance(value, dict): raise ValueError('Missing native render fraction')
    return Fraction(integer(value.get('numerator')), integer(value.get('denominator'), minimum=1))


def frame_fraction(frame):
    if not isinstance(frame, dict) or frame.get('type') not in ('FrameNumber', 'ConstantFramerateFrameNumber'):
        raise ValueError('Missing native render-product frame identifier')
    integer(frame.get('frameNumber'))
    return fraction({'numerator':frame.get('rationalTimeOfSimNumerator'),
                     'denominator':frame.get('rationalTimeOfSimDenominator')})


def validate_batch(batch, *, matched=True):
    """Baseline may straddle batches; an accepted capture may not."""
    if not isinstance(batch, dict) or set(batch.get('cameras',{})) != VIEWS:
        raise ValueError('Missing render-product batch evidence')
    scheduled, completed = fraction(batch.get('scheduled')), fraction(batch.get('completed'))
    paths, nodes = set(), set()
    for camera in batch['cameras'].values():
        path = camera.get('render_product')
        if not isinstance(path, str) or not path.startswith('/') or path in paths:
            raise ValueError('Distinct existing render products required')
        paths.add(path)
        stamp = frame_fraction(camera.get('frame'))
        dispatch = camera.get('dispatch',{})
        if (dispatch.get('input_product') != path or dispatch.get('output_product') != path or
                dispatch.get('type') != 'omni.syntheticdata.SdOnNewRenderProductFrame' or
                camera.get('frame_node_type') != 'omni.syntheticdata.SdFrameIdentifier' or
                camera.get('exec_source') != dispatch.get('node')):
            raise ValueError('Frame marker is not bound to its render product')
        bindings = camera.get('bindings',{})
        if set(bindings) != {'rgb','depth_linear'}:
            raise ValueError('Both RGB and depth bindings required')
        for binding in (dispatch, {'node':camera.get('frame_node')}, *bindings.values()):
            node = binding.get('node')
            if not isinstance(node,str) or not node.startswith('/') or node in nodes:
                raise ValueError('Distinct native camera graph nodes required')
            nodes.add(node)
        if any(v.get('render_products') != [path] for v in bindings.values()):
            raise ValueError('RGB/depth annotator attached to a different render product')
        if matched and stamp != scheduled:
            raise ValueError('Stale or mixed render-product batch')
    if matched and completed != scheduled:
        raise ValueError('Scheduled and completed render batches differ')


def validate_advance(batch, baseline):
    validate_batch(batch)
    validate_batch(baseline, matched=False)
    # force-Fabric render happens BEFORE baseline. No offset, float tolerance,
    # image-difference heuristic or equality to physics time is permitted.
    newest = max(fraction(baseline['scheduled']), fraction(baseline['completed']),
                 *(frame_fraction(c['frame']) for c in baseline['cameras'].values()))
    if fraction(batch['scheduled']) <= newest:
        raise ValueError('Render batch did not advance beyond the pre-capture baseline')
    for view, camera in batch['cameras'].items():
        old = baseline['cameras'][view]
        for key in ('render_product','frame_node','frame_node_type','exec_source','dispatch','bindings'):
            if camera[key] != old[key]: raise ValueError('Native camera binding changed during capture')
        if camera['frame']['frameNumber'] <= old['frame']['frameNumber']:
            raise ValueError('Render-product frame number did not advance')


def make_receipts(batch, clock, capture):
    return {view:{'version':'render_batch_v2', 'capture':capture,
                  'scheduled':batch['scheduled'], 'completed':batch['completed'],
                  'camera':camera, **clock, 'wait_for_render':True,
                  'physics_ticks_in_capture':0, 'render_batch_verified':True}
            for view,camera in batch['cameras'].items()}


def register_frame_annotator(rep, synthetic):
    # A private, uniquely named native node; no Python counter is passed off as
    # proof of a completed frame. Registration alone does not author USD.
    if FRAME_ANNOTATOR in synthetic._ogn_templates_registry:
        raise ValueError('Render-batch annotator already registered; refusing to overwrite')
    rep.AnnotatorRegistry.register_annotator_from_node(
        name=FRAME_ANNOTATOR, node_type_id='omni.syntheticdata.SdFrameIdentifier',
        input_rendervars=[synthetic.NodeConnectionTemplate('PostProcessDispatch',
            attributes_mapping={'outputs:exec':'inputs:exec', 'outputs:renderResults':'inputs:renderResults'})],
        output_data_type=None, output_is_2d=False)


def native_batch(sensors, references, orchestrator, graph):
    """Read only already-existing native metadata; never evaluate/update graph."""
    dispatcher = graph.get_node_by_path('/Render/PostProcess/SDGPipeline/PostProcessDispatcher')
    if not dispatcher: raise ValueError('No native completion dispatcher')
    def boundaries():
        scheduled = orchestrator._orchestrator._sim_times_to_write
        if not scheduled: raise ValueError('No native scheduled render batch')
        pair = scheduled[-1]
        if len(pair) != 2: raise ValueError('Malformed native scheduled render batch')
        return {'scheduled':{'numerator':pair[0], 'denominator':pair[1]},
                'completed':{key:dispatcher.get_attribute('outputs:referenceTime'+name).get()
                             for key,name in (('numerator','Numerator'),('denominator','Denominator'))}}
    batch = {**boundaries(), 'cameras':{}}
    frame_nodes = {}
    for view,sensor in sensors.items():
        product = sensor.render_product
        path = product if isinstance(product,str) else product.path
        node = references[view].get_node()
        frame_nodes[view] = node
        if node.get_type_name() != 'omni.syntheticdata.SdFrameIdentifier':
            raise ValueError('Frame marker is not the native frame identifier')
        connections = node.get_attribute('inputs:renderResults').get_upstream_connections()
        if len(connections) != 1: raise ValueError('Ambiguous native frame source')
        upstream = connections[0].get_node()
        # Inspect the actual native connection, not a constructed path label.
        if upstream.get_type_name() != 'omni.syntheticdata.SdOnNewRenderProductFrame':
            raise ValueError('Frame marker source is not a per-product dispatcher')
        execution = node.get_attribute('inputs:exec').get_upstream_connections()
        if (len(execution)!=1 or str(execution[0].get_node().get_prim_path())!=str(upstream.get_prim_path()) or
                connections[0].get_name()!='outputs:renderResults' or execution[0].get_name()!='outputs:exec'):
            raise ValueError('Frame exec and renderResults must use the same product dispatcher')
        bindings = {}
        for modality in ('rgb','depth_linear'):
            anno = sensor._annotators[modality]
            bindings[modality] = {'render_products':list(anno._render_products),
                                 'node':str(anno.get_node().get_prim_path())}
        batch['cameras'][view] = {'render_product':path, 'frame_node':str(node.get_prim_path()),
            'frame_node_type':node.get_type_name(), 'exec_source':str(upstream.get_prim_path()),
            'frame':{key:node.get_attribute('outputs:'+key).get() for key in FRAME_FIELDS},
            'dispatch':{'node':str(upstream.get_prim_path()), 'type':upstream.get_type_name(),
                        'input_product':upstream.get_attribute('inputs:renderProductPath').get(),
                        'output_product':upstream.get_attribute('outputs:renderProductPath').get()},
            'bindings':bindings}
    for view,node in frame_nodes.items():
        if {key:node.get_attribute('outputs:'+key).get() for key in FRAME_FIELDS} != batch['cameras'][view]['frame']:
            raise ValueError('Render-product frame changed during metadata read')
    if boundaries() != {k:batch[k] for k in ('scheduled','completed')}:
        raise ValueError('Native render batch changed during metadata read')
    return batch
