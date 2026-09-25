from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

from render_batch_fixture import batch_fixture
from semantic_robot.v2.render_batch import (validate_batch,validate_advance,fraction,
    register_frame_annotator,FRAME_ANNOTATOR,FRAME_FIELDS,native_batch)


class RenderBatchTests(unittest.TestCase):
    def native_fixture(self):
        batch=batch_fixture()
        def node(path,kind,values):
            result=Mock()
            result.get_prim_path.return_value=path;result.get_type_name.return_value=kind
            attrs={name:Mock(get=Mock(return_value=value)) for name,value in values.items()}
            result.get_attribute.side_effect=attrs.__getitem__
            return result,attrs
        dispatcher,_=node('/dispatcher','global',{
            'outputs:referenceTimeNumerator':132,'outputs:referenceTimeDenominator':30})
        sensors,references={},{}
        for view,c in batch['cameras'].items():
            upstream,attrs=node(c['dispatch']['node'],c['dispatch']['type'],{
                'inputs:renderProductPath':c['render_product'],'outputs:renderProductPath':c['render_product']})
            frame,attrs=node(c['frame_node'],c['frame_node_type'],{'outputs:'+k:v for k,v in c['frame'].items()})
            for target in ('exec','renderResults'):
                connection=Mock();connection.get_node.return_value=upstream
                connection.get_name.return_value='outputs:'+target
                attr=Mock();attr.get_upstream_connections.return_value=[connection]
                attrs['inputs:'+target]=attr
            references[view]=Mock();references[view].get_node.return_value=frame
            annos={}
            for modality,binding in c['bindings'].items():
                annos[modality]=Mock();annos[modality]._render_products=[c['render_product']]
                annos[modality].get_node.return_value=Mock(get_prim_path=Mock(return_value=binding['node']))
            sensors[view]=SimpleNamespace(render_product=SimpleNamespace(path=c['render_product']),_annotators=annos)
        orchestrator=SimpleNamespace(_orchestrator=SimpleNamespace(_sim_times_to_write=[(132,30)]))
        graph=Mock();graph.get_node_by_path.return_value=dispatcher
        return sensors,references,orchestrator,graph

    def test_native_reader_uses_actual_connections_and_does_not_sample_pixels_or_update_graph(self):
        sensors,references,orchestrator,graph=self.native_fixture()
        result=native_batch(sensors,references,orchestrator,graph)
        self.assertEqual(result,batch_fixture());validate_batch(result)
        for sensor in sensors.values():
            for anno in sensor._annotators.values():anno.get_data.assert_not_called()
        graph.evaluate.assert_not_called()

    def test_native_reader_rejects_wrong_types_or_cross_product_exec(self):
        for mode in ('frame_type','dispatch_type','exec_node','exec_port','results_port','multiple'):
            sensors,refs,orch,graph=self.native_fixture()
            frame=refs['head'].get_node()
            results=frame.get_attribute('inputs:renderResults').get_upstream_connections()[0]
            execution=frame.get_attribute('inputs:exec').get_upstream_connections()[0]
            if mode=='frame_type':frame.get_type_name.return_value='wrong'
            if mode=='dispatch_type':results.get_node().get_type_name.return_value='global'
            if mode=='exec_node':execution.get_node.return_value=Mock(get_prim_path=Mock(return_value='/wrong'))
            if mode=='exec_port':execution.get_name.return_value='outputs:wrong'
            if mode=='results_port':results.get_name.return_value='outputs:wrong'
            if mode=='multiple':frame.get_attribute('inputs:exec').get_upstream_connections.return_value=[]
            with self.subTest(mode=mode),self.assertRaises(ValueError):native_batch(sensors,refs,orch,graph)

    def test_native_reader_brackets_global_schedule_and_each_product_marker(self):
        for mode in ('global','scheduled',*('camera_'+view for view in ('head','left_wrist','right_wrist'))):
            sensors,refs,orch,graph=self.native_fixture()
            if mode=='global':
                graph.get_node_by_path().get_attribute('outputs:referenceTimeNumerator').get.side_effect=[132,133]
            elif mode=='scheduled':
                field=refs['head'].get_node().get_attribute('outputs:frameNumber')
                def mutate():
                    orch._orchestrator._sim_times_to_write.append((133,30));return 132
                field.get.side_effect=mutate
            else:
                field=refs[mode.removeprefix('camera_')].get_node().get_attribute('outputs:frameNumber')
                field.get.side_effect=[132,133]
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'changed during metadata'):
                native_batch(sensors,refs,orch,graph)

    def test_exact_rational_native_integers_and_distinct_time_domain(self):
        batch=batch_fixture();batch['completed']={'numerator':np.uint64(22),'denominator':np.uint64(5)}
        validate_batch(batch);validate_advance(batch,batch_fixture(131))
        self.assertEqual(fraction(batch['completed']),fraction(batch['scheduled']))
        for n,d in ((True,1),(1.,1),(-1,1),(1,0),(1,False)):
            with self.subTest(n=n,d=d),self.assertRaises(ValueError):fraction({'numerator':n,'denominator':d})

    def test_every_view_must_have_real_frame_and_both_correct_bindings(self):
        for view in ('head','left_wrist','right_wrist'):
            changes=[lambda b:b['cameras'][view]['frame'].update(type='NoFrameNumber'),
                     lambda b:b['cameras'][view]['frame'].update(frameNumber=-1),
                     lambda b:b['cameras'][view]['frame'].update(rationalTimeOfSimNumerator=131),
                     lambda b:b['cameras'][view]['bindings']['rgb'].update(render_products=['/wrong']),
                     lambda b:b['cameras'][view]['bindings'].pop('depth_linear'),
                     lambda b:b['cameras'][view]['dispatch'].update(output_product='/wrong'),
                     lambda b:b['cameras'][view].update(frame_node='/Render/head/dispatch')]
            for change in changes:
                batch=batch_fixture();change(batch)
                with self.subTest(view=view,change=change),self.assertRaises(ValueError):validate_batch(batch)

    def test_global_match_cannot_cover_one_stale_camera_and_no_counter_fallback(self):
        a,b=batch_fixture(132),batch_fixture(133)
        validate_advance(b,a)
        for change in (lambda b:b.update(completed={'numerator':134,'denominator':30}),
                       lambda b:b['cameras']['head']['frame'].update(frameNumber=132),
                       lambda b:b['cameras']['head']['frame'].update(rationalTimeOfSimNumerator=132),
                       lambda b:b['cameras']['head']['bindings']['rgb'].update(node='/different')):
            bad=deepcopy(b);change(bad)
            with self.assertRaises(ValueError):validate_advance(bad,a)
        a['completed']={'numerator':133,'denominator':30}
        with self.assertRaisesRegex(ValueError,'baseline'):validate_advance(b,a)

    def test_register_uses_native_per_product_dispatch_not_global_reference(self):
        rep=SimpleNamespace(AnnotatorRegistry=Mock())
        synthetic=SimpleNamespace(_ogn_templates_registry={},NodeConnectionTemplate=Mock(return_value='connection'))
        register_frame_annotator(rep,synthetic)
        synthetic.NodeConnectionTemplate.assert_called_once_with('PostProcessDispatch',
            attributes_mapping={'outputs:exec':'inputs:exec','outputs:renderResults':'inputs:renderResults'})
        args=rep.AnnotatorRegistry.register_annotator_from_node.call_args.kwargs
        self.assertEqual(args['node_type_id'],'omni.syntheticdata.SdFrameIdentifier')
        self.assertEqual(args['name'],FRAME_ANNOTATOR)
        synthetic._ogn_templates_registry[FRAME_ANNOTATOR]=object()
        with self.assertRaises(ValueError):register_frame_annotator(rep,synthetic)


if __name__=='__main__':unittest.main()
