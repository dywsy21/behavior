"""Freeze signed two-wave calibration anchors before any observer prediction."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from recovery_sft_data import require_training_pool
from recovery_prospective_adapter import pool_sources,make_pool_selection,pool_config_keys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['root'])
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Use new preselection output and clean source')
    if cfg['schema']!='prospective_observer_adapter_calibration_v1':raise ValueError('Explicit prospective recipe required')
    def read_bound(path,sha):
        path=root/path
        if file_sha(path)!=sha:raise ValueError('Changed pinned input: '+str(path))
        return json.loads(path.read_text())
    pool=read_bound(cfg['source_pool'],cfg['source_pool_sha256']);cohorts=[]
    keys=pool_config_keys(pool)
    if len(cfg['cohorts'])!=len(keys):raise ValueError('Every declared collection wave required')
    for index,(item,key) in enumerate(zip(cfg['cohorts'],keys)):
        source_cfg=REPO/pool[key];source_sha=file_sha(source_cfg)
        cohort=read_bound(item['path'],item['sha256'])
        if ((pool['schema']!='prospective_calibration_source_pool_v4' or index==2)
                and json.loads(source_cfg.read_text())['selected_observer']['fit_result_sha256']!=cfg['fit_result_sha256']):
            raise ValueError('Fit must have been fixed before source collection')
        cohorts.append((cohort,item['sha256'],source_sha))
    fit=read_bound(cfg['fit_result'],cfg['fit_result_sha256'])
    fit_cfg=read_bound(cfg['fit_config'],fit['config_sha256'])
    if (fit['selected_checkpoint_sha256']!=cfg['selected_observer_sha256']
            or (pool.get('fit_admission_sha256') is not None
                and pool['fit_admission_sha256']!=fit_cfg['admission_sha256'])):
        raise ValueError('Exposure must come from the exact preselected model fit')
    _,old=require_training_pool(root/fit_cfg['admission'],'outcome',fit_cfg['admission_sha256'])
    exposed={r['candidate']['source_group'] for r in old}
    if pool['schema']=='prospective_calibration_source_pool_v4':
        from recovery_reserved_calibration import load_reserved_pool
        provenance=load_reserved_pool(root,REPO,pool,cohorts,exposed)
    else:provenance=pool_sources(pool,cohorts,exposed_groups=exposed)
    receipt,rows=require_training_pool(root/cfg['admission'],'outcome',cfg['admission_sha256'],purpose='calibration')
    partition=json.loads((root/cfg['admission']/receipt['evaluation_partition_file']).read_text())
    if (partition.get('schema')!='recovery_evaluation_partition_v3'
            or partition['cohort_sha256']!=cfg['source_pool_sha256']
            or partition['source_cohort_by_group']!=provenance['source_cohort_by_group']
            or partition['reserved_frozen_test_groups']!=provenance['reserved_frozen_test_groups']):
        raise ValueError('Union must retain each original source role/signature')
    selection=make_pool_selection(pool,provenance,rows,pool_sha256=cfg['source_pool_sha256'],
        fit_result_sha256=cfg['fit_result_sha256'])
    selection.update(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        admission_sha256=cfg['admission_sha256'],cohort_file_sha256=[v[1] for v in cohorts])
    with a.output.open('x') as stream:stream.write(json.dumps(selection,indent=2)+'\n')
    print(json.dumps(dict(status='anchors_frozen_before_predictions',sha256=file_sha(a.output),
        declared=selection['declared_source_count'],selected=selection['calibration_source_count'],
        unavailable=selection['unavailable_source_count'],reserved=len(selection['reserved_groups']),
        ineligible_previously_exposed=selection.get('ineligible_source_count',0),
        previously_predicted_sources=selection.get('previously_predicted_source_count',0),
        optimizer_updates=0,calibration_fitted=False)))


if __name__=='__main__':main()
