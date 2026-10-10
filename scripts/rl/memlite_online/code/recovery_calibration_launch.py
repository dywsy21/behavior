"""Prepare the pinned observer calibration chain without running any model.

This composes the existing full union audit and prediction-blind selection
tools. It does not replace their validators, select a checkpoint, change gates,
read the reserved test media, start GPU jobs, or grant deployment permission.
"""
import json
import os
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys

from recovery_corpus import file_sha


def child(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Require a root-relative, nonempty dependency/output path')
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError('Path escapes its declared data/code root')
    return path


def bound(root, reference):
    if (set(reference) != {'path', 'sha256'}
            or not re.fullmatch(r'[0-9a-f]{64}', reference['sha256'])):
        raise ValueError('Dependency must have an exact path and immutable SHA256')
    path = child(root, reference['path'])
    if not path.is_file() or file_sha(path) != reference['sha256']:
        raise ValueError('Missing/changed pinned dependency: ' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def check_transport(directory, manifest_sha256, admission_sha256):
    directory = Path(directory).resolve()
    path = bound(directory, dict(path='transfer-manifest.json', sha256=manifest_sha256))
    manifest = read(path)
    if (manifest.get('schema') != 'recovery_release_transfer_v1'
            or manifest.get('partial_reviewed_unit') is not True
            or manifest.get('grants_training_launch_permission') is not False
            or manifest.get('all_three_pools_ready') is not False
            or manifest.get('contains_weights') is not False
            or manifest.get('contains_credentials') is not False
            or manifest.get('admission_sha256') != admission_sha256):
        raise ValueError('Only the independently signed calibration-only transport is allowed')
    seen = set()
    for item in manifest['files']:
        file = bound(directory, dict(path=item['path'], sha256=item['sha256']))
        if file in seen or file.stat().st_size != item['bytes']:
            raise ValueError('Repeated/changed transferred file')
        seen.add(file)
    if not seen:
        raise ValueError('Empty transport')
    return dict(files=len(seen), bytes=sum(item['bytes'] for item in manifest['files']),
        manifest_sha256=manifest_sha256, admission_sha256=admission_sha256)


def preflight(declaration, repo):
    if declaration.get('schema') != 'observer_reserved_calibration_launch_v1':
        raise ValueError('Explicit immutable reserved-calibration declaration required')
    root, repo = Path(declaration['root']).resolve(), Path(repo).resolve()
    if not root.is_dir() or root not in repo.parents:
        raise ValueError('Frozen source must be inside the shared experiment root')
    union_path = bound(repo, declaration['union_spec'])
    pool_path = bound(repo, declaration['source_pool'])
    union, pool = read(union_path), read(pool_path)
    fit_path = bound(root, declaration['fit_result'])
    fit = read(fit_path)
    fit_config_path = bound(root, declaration['fit_config'])
    fit_config = read(fit_config_path)
    weight_path = bound(root, declaration['selected_observer'])
    if (pool.get('schema') != 'prospective_calibration_source_pool_v4'
            or fit.get('status') != 'observer_adapter_fit_complete_not_deployed'
            or fit['config_sha256'] != declaration['fit_config']['sha256']
            or fit['frozen_before_sha256'] != fit['frozen_after_sha256']
            or fit['selected_checkpoint_sha256'] != declaration['selected_observer']['sha256']
            or pool['selected_observer_sha256'] != fit['selected_checkpoint_sha256']
            or pool['fit_result_sha256'] != declaration['fit_result']['sha256']
            or pool['fit_admission_sha256'] != fit_config['admission_sha256']
            or pool['high_sha256'] != fit_config['high']['sha256']
            or Path(fit_config['root']).resolve() != root):
        raise ValueError('Completed fit, original backbone or predeclared model changed')
    bound(root, fit_config['high'])
    bound(root, dict(path=fit_config['admission'] + '/admission.json',
        sha256=fit_config['admission_sha256']))
    cohorts = declaration['cohorts']
    if len(cohorts) != 3 or len({item['path'] for item in cohorts}) != 3:
        raise ValueError('Exactly the two old signed cohorts and the fresh third wave are required')
    for item in cohorts:
        bound(root, item)
    transport = declaration['new_unit']
    transport_root = child(root, transport['directory'])
    transport_receipt = check_transport(transport_root, transport['manifest_sha256'],
        transport['admission_sha256'])
    new_admission = bound(root, dict(path=transport['admission'], sha256=transport['admission_sha256']))
    if (transport_root not in new_admission.parents
            or Path(union['evidence_root']).resolve() != root
            or union['evaluation_partition']['base_unit_count'] != 0
            or union['evaluation_partition']['cohort_sha256'] != declaration['source_pool']['sha256']
            or union.get('prediction_exposure_quarantine') != pool['prior_prediction_exposure']
            or Path(union['units'][-1]['corpus']).resolve() / 'admission/admission.json' != new_admission):
        raise ValueError('Transport, signed pool or prior-prediction quarantine mismatch')
    run, corpus = child(root, declaration['run_output']), child(root, declaration['corpus_output'])
    if (not declaration['run_output'].startswith('runs/')
            or not declaration['corpus_output'].startswith('datasets/')
            or len(Path(declaration['run_output']).parts) < 2
            or len(Path(declaration['corpus_output']).parts) < 2
            or run == corpus or run in corpus.parents or corpus in run.parents
            or run.exists() or corpus.exists()):
        raise ValueError('Use independent new run/corpus paths; preserve prior partial attempts')
    config = dict(schema='prospective_observer_adapter_calibration_v1',
        owner=declaration['owner'], root=str(root),
        source_pool=str(pool_path.relative_to(root)),
        source_pool_sha256=declaration['source_pool']['sha256'], cohorts=cohorts,
        fit_result=str(fit_path.relative_to(root)), fit_result_sha256=declaration['fit_result']['sha256'],
        fit_config=str(fit_config_path.relative_to(root)),
        selected_observer=str(weight_path.relative_to(root)),
        selected_observer_sha256=declaration['selected_observer']['sha256'],
        high_sha256=pool['high_sha256'], corpus=str(corpus.relative_to(root)),
        admission=str((corpus / 'admission').relative_to(root)),
        interpretation='Fixed v11; all217 attempts accounted, old90 predicted groups excluded, '
            'original phase quotas/safety gates and old20 test roles unchanged. No model selection or deployment.')
    return dict(root=root, repo=repo, run=run, corpus=corpus, union_path=union_path,
        config=config, transport=transport_receipt)


def prepare(declaration, repo, *, source_commit, declaration_sha256, runner=subprocess.run):
    if (not re.fullmatch(r'[0-9a-f]{40}', source_commit)
            or not re.fullmatch(r'[0-9a-f]{64}', declaration_sha256)):
        raise ValueError('Actual frozen source/declaration identity required')
    checked = preflight(declaration, repo)
    root, repo, run, corpus = (checked[k] for k in ('root', 'repo', 'run', 'corpus'))
    run.mkdir(parents=True)
    provenance = dict(source_commit=source_commit, declaration_sha256=declaration_sha256,
        pid=os.getpid(), transport=checked['transport'], optimizer_updates=0,
        model_forwards=0, runtime_deployed=False)
    write_new(run / 'preflight.json', dict(provenance, status='dependencies_verified_before_union'))
    try:
        # Existing full archive/approval/exposure audits remain authoritative.
        runner([sys.executable, str(repo/'scripts/rl/memlite_online/tools/merge_reviewed_recovery.py'),
            '--spec', str(checked['union_path']), '--output', str(corpus)], check=True, cwd=repo)
        receipt = read(corpus/'receipt.json')
        admission = corpus/'admission/admission.json'
        if receipt.get('status') != 'passed' or receipt['admission_sha256'] != file_sha(admission):
            raise ValueError('Merged corpus did not produce a valid full-read receipt')
        config = dict(checked['config'], admission_sha256=file_sha(admission))
        before = run/'preselection-config.json'
        write_new(before, config)
        anchors = run/'anchor-selection.json'
        runner([sys.executable, str(repo/'scripts/rl/memlite_online/tools/prepare_observer_calibration_pool.py'),
            '--config', str(before), '--output', str(anchors)], check=True, cwd=repo)
        selection = read(anchors)
        if (selection.get('schema') != 'independent_calibration_pool_selection_v4'
                or selection.get('declared_before_model_predictions') is not True
                or selection['admission_sha256'] != config['admission_sha256']
                or selection['source_pool_sha256'] != config['source_pool_sha256']
                or selection['selected_observer_sha256'] != config['selected_observer_sha256']
                or selection['fit_result_sha256'] != config['fit_result_sha256']
                or selection['source_commit'] != source_commit
                or len(selection['selected_groups']) != 90
                or len(set(selection['selected_groups'])) != 90
                or len(selection['selected_anchors']) != 90
                or len({r['sample_id'] for r in selection['selected_anchors']}) != 90
                or {r['source_group'] for r in selection['selected_anchors']} != set(selection['selected_groups'])
                or Counter(r['value'] for r in selection['selected_anchors']) !=
                    Counter(IN_PROGRESS=30, SUCCEEDED=30, FAILED=30)):
            raise ValueError('Original preselection tool did not close the exact independent90')
        final = run/'calibration-config.json'
        write_new(final, dict(config, anchor_selection=str(anchors.relative_to(root)),
            anchor_selection_sha256=file_sha(anchors)))
        result = dict(provenance, status='calibration_inputs_prepared_no_gpu_or_deployment',
            corpus_receipt_sha256=file_sha(corpus/'receipt.json'),
            admission_sha256=file_sha(admission), anchor_selection_sha256=file_sha(anchors),
            config=str(final), config_sha256=file_sha(final), selected_groups=90,
            declared_sources=selection['declared_source_count'],
            unavailable_sources=selection['unavailable_source_count'],
            previously_predicted_sources=selection['previously_predicted_source_count'],
            calibration_command=[sys.executable, str(repo/'scripts/eval/calibrate_recovery_observer_adapter.py'),
                '--config', str(final), '--output', str(run/'calibration')],
            requires_separate_idle_gpu_preflight=True, goal_complete=False)
        write_new(run/'result.json', result)
        return result
    except BaseException as error:
        write_new(run/'failure.json', dict(provenance, status='preparation_failed_no_gpu_started',
            error=repr(error), partial_run_and_corpus_preserved=True))
        raise
