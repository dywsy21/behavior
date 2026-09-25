"""CPU-only native batch fixture; render time intentionally != physics time."""
from copy import deepcopy


def batch_fixture(number=132):
    stamp={'numerator':number,'denominator':30}
    cameras={}
    for view in ('head','left_wrist','right_wrist'):
        path='/Render/'+view
        cameras[view]={'render_product':path,'frame_node':path+'/frame',
            'frame_node_type':'omni.syntheticdata.SdFrameIdentifier','exec_source':path+'/dispatch',
            'frame':{'type':'ConstantFramerateFrameNumber','frameNumber':number,
                     'rationalTimeOfSimNumerator':number,'rationalTimeOfSimDenominator':30},
            'dispatch':{'node':path+'/dispatch','type':'omni.syntheticdata.SdOnNewRenderProductFrame',
                        'input_product':path,'output_product':path},
            'bindings':{modality:{'node':path+'/'+modality,'render_products':[path]}
                        for modality in ('rgb','depth_linear')}}
    return {'scheduled':stamp,'completed':deepcopy(stamp),'cameras':cameras}


def camera_receipts(times):
    return {view:{'freshness_proof':'native_render_batch_v2','native_time':time,
                  'rgb_sha256':'a'*64,'depth_sha256':'b'*64} for view,time in times.items()}
