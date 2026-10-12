import importlib.util
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

FILE=Path(__file__).resolve().parents[1]/'scripts/eval/memlite_sft100/gpu_start_guard.py'
spec=importlib.util.spec_from_file_location('gpu_guard',FILE)
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


def xml(rows=(), free=70000):
    return '<nvidia_smi_log><gpu><fb_memory_usage><free>'+str(free)+' MiB</free></fb_memory_usage><processes>'+''.join(
        f'<process_info><pid>{pid}</pid><used_memory>{size} MiB</used_memory></process_info>'
        for pid,size in rows)+'</processes></gpu></nvidia_smi_log>'


class GuardTests(unittest.TestCase):
    def test_exact_peers_only_memory_and_reserve_fail_closed(self):
        guard.validate_snapshot(xml(),set())
        result=guard.validate_snapshot(xml([(11,174),(12,174),(11,174)]),{11,12})
        self.assertEqual(result['observed_peer_mib'],{11:174,12:174})
        for data,pids in [(xml([(13,1)]),{11,12}),(xml([(11,513)]),{11}),
                          (xml(free=24000),set()),(xml([(11,1)]).replace('1 MiB','N/A'),{11})]:
            with self.assertRaises(ValueError):guard.validate_snapshot(data,pids)

    def test_peer_binds_config_case_command_output_environment_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);proc=root/'proc/123';proc.mkdir(parents=True)
            config=root/'config.json';status=root/'run/status.json';status.parent.mkdir()
            cfg={'instance_id':301,'cases':{'radio':{'sim_gpu':0,'task':'radio'}}}
            config.write_text(json.dumps(cfg))
            receipt=dict(case='radio',task='radio',status='running',instance_id=301,pid=123,
                config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),source_commit='a'*40)
            status.write_text(json.dumps(receipt))
            args=['python','/source/scripts/eval/run_causal_full_episode.py','--config',str(config),
                  '--case','radio','--output',str(status.parent)]
            (proc/'cmdline').write_bytes(b'\0'.join(x.encode() for x in args)+b'\0')
            (proc/'environ').write_bytes(b'OMNIGIBSON_GPU_ID=0\0')
            with patch.object(guard.subprocess,'check_output',return_value='a'*40+'\n'):
                self.assertEqual(guard.peer_identity(status,config_path=config,gpu=2,proc_root=root/'proc'),123)
                with self.assertRaises(ValueError):guard.peer_identity(status,config_path=config,gpu=0,proc_root=root/'proc')
                (proc/'environ').write_bytes(b'OMNIGIBSON_GPU_ID=2\0')
                with self.assertRaises(ValueError):guard.peer_identity(status,config_path=config,gpu=2,proc_root=root/'proc')
                (proc/'environ').write_bytes(b'OMNIGIBSON_GPU_ID=0\0')
                receipt['source_commit']='b'*40;status.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):guard.peer_identity(status,config_path=config,gpu=2,proc_root=root/'proc')


if __name__=='__main__':unittest.main()
