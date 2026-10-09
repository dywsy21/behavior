"""Pinned post2 light-synchronizer cache AND actual USD visibility snapshot.

World snapshots alone do not preserve the synchronizer's edge history. Never
call reset_from_current_state while restoring: it turns all lights on. This
adapter is collection-only, and all fields stay outside policy observations.
"""
from copy import deepcopy

ROOM = {'downlight', 'room_light', 'wall_mounted_light'}
SELF = {'floor_lamp', 'table_lamp'}


def _booleans(values):
    if not isinstance(values, dict): raise ValueError('Light values must be a mapping')
    result = {}
    for key, value in values.items():
        if hasattr(value, 'item'): value = value.item()
        if not isinstance(key, str) or not key or type(value) is not bool:
            raise ValueError('Invalid light identity or Boolean value')
        result[key] = value
    return result


def _objects(sync, scene):
    if (type(sync).__name__ != 'LightToggleSynchronizer' or sync.scene is not scene
            or set(vars(sync)) != {'scene', '_toggle_values', '_target_visibility'}):
        raise ValueError('Unregistered light-synchronizer state or scene')
    objects = {o.name: o for o in scene.objects if o.category in ROOM | SELF | {'electric_switch'}}
    topology = {name: dict(category=o.category, rooms=sorted(o.in_rooms or [])) for name, o in objects.items()}
    return objects, topology


def _self_visibility(obj):
    from omnigibson.eval.utils.light_utils import _iter_descendant_prims
    import omnigibson.lazy as lazy
    return {str(prim.GetPath()): str(lazy.pxr.UsdGeom.Imageable(prim).GetVisibilityAttr().Get())
            for prim in _iter_descendant_prims(obj.prim) if 'Light' in prim.GetTypeName()}


def capture_lights(inst):
    sync = inst.light_synchronizer
    if sync is None: return None
    objects, topology = _objects(sync, inst.env_accessor.scene)
    toggles = _booleans(sync._toggle_values); visibility = _booleans(sync._target_visibility)
    expected = {n for n, o in objects.items() if o.category in ROOM | SELF}
    if set(visibility) != expected or toggles != _booleans(sync._get_toggle_values()):
        raise ValueError('Uninitialized or stale synchronizer; snapshot only after actual sync')
    prims = {}
    for name in expected:
        obj = objects[name]
        if obj.category in ROOM:
            if bool(obj.visible) != visibility[name]: raise ValueError('Room light render state differs from cache')
        else:
            prims[name] = _self_visibility(obj)
            token = 'inherited' if visibility[name] else 'invisible'
            if any(value != token for value in prims[name].values()):
                raise ValueError('Self-light USD visibility differs from tracked state')
    return dict(schema='post2_light_branch_state_v1', topology=topology, toggle_values=toggles,
                target_visibility=visibility, self_prim_visibility=prims)


def validate_light_restore(inst, saved):
    sync = inst.light_synchronizer
    if sync is None:
        if saved is not None: raise ValueError('Saved light state applied to a scene without synchronizer')
        return
    if saved is None: raise ValueError('Missing light state; cannot silently reset illumination')
    objects, topology = _objects(sync, inst.env_accessor.scene)
    if (set(saved) != {'schema', 'topology', 'toggle_values', 'target_visibility', 'self_prim_visibility'}
            or saved['schema'] != 'post2_light_branch_state_v1' or saved['topology'] != topology
            or _booleans(saved['toggle_values']) != _booleans(sync._get_toggle_values())):
        raise ValueError('Wrong lighting topology or world toggles not restored before metadata')
    visibility = _booleans(saved['target_visibility'])
    if (set(visibility) != {n for n, o in objects.items() if o.category in ROOM | SELF}
            or set(saved['self_prim_visibility']) != {n for n, o in objects.items() if o.category in SELF}):
        raise ValueError('Missing light render targets')
    for name, values in saved['self_prim_visibility'].items():
        if (set(values) != set(_self_visibility(objects[name]))
                or any(v != ('inherited' if visibility[name] else 'invisible') for v in values.values())):
            raise ValueError('Changed light prim topology or invalid saved visibility')


def _apply_visibility(obj, visible):
    if obj.category in ROOM: obj.visible = visible
    else:
        from omnigibson.eval.utils.light_utils import _set_light_prims_visible
        _set_light_prims_visible(obj, visible)


def restore_lights(inst, saved):
    validate_light_restore(inst, saved)
    if saved is None: return
    sync = inst.light_synchronizer
    objects, _ = _objects(sync, inst.env_accessor.scene)
    for name, visible in saved['target_visibility'].items(): _apply_visibility(objects[name], visible)
    sync._toggle_values = deepcopy(saved['toggle_values'])
    sync._target_visibility = deepcopy(saved['target_visibility'])
    if capture_lights(inst) != saved: raise ValueError('Light state did not round-trip')
