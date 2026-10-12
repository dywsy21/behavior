"""Exact-byte CPU transport for the registered TRAIN diagnostic, not training."""
import copy
import json
from pathlib import Path
import shutil

from recovery_calibration_launch import bound, read, write_new
from recovery_corpus import digest, file_sha
from recovery_handoff_inference import tree_sha


def portable_arch(arch, root):
    result = copy.deepcopy(arch); root = Path(root)
    if (result['hf_processor_path'] != str(root/'models/qwen3_5_2b_base_processor')
            or result['AT_CONFIG']['ckpt_dir'] != str(root/'models/action_tokenizer.pt')):
        raise ValueError('Unknown model dependency relocation')
    result['hf_processor_path'] = 'assets/hf'
    result['AT_CONFIG']['ckpt_dir'] = 'assets/action_tokenizer.pt'
    def check(value):
        if isinstance(value,dict):
            for item in value.values(): check(item)
        elif isinstance(value,(list,tuple)):
            for item in value: check(item)
        elif isinstance(value,str) and value.startswith('/'):
            raise ValueError('Unregistered absolute dependency in portable architecture')
    check(result)
    return result


def materialize_arch(arch, root):
    result = copy.deepcopy(arch)
    if result['hf_processor_path'] != 'assets/hf' or result['AT_CONFIG']['ckpt_dir'] != 'assets/action_tokenizer.pt':
        raise ValueError('Changed portable architecture paths')
    result['hf_processor_path'] = str(Path(root)/'assets/hf')
    result['AT_CONFIG']['ckpt_dir'] = str(Path(root)/'assets/action_tokenizer.pt')
    return result


def check_prefix_rows(rows, accepted):
    if (accepted['status'] != 'passed' or accepted['data_split'] != 'train'
            or accepted['optimizer_updates'] != 0 or len(rows) != 383
            or rows != accepted['rows'] or len({r['sample_id'] for r in rows}) != 383):
        raise ValueError('Portable inputs differ from the complete original TRAIN CPU acceptance')


def export_pack(output, *, registration, registration_sha256, diagnostic, diagnostic_sha256,
                model_config, prepared_rows, cpu_rows, common, repo):
    import torch
    from recovery_handoff_heldout import validate_fixed_fit
    root = Path(diagnostic['root']); output = Path(output)
    if (registration['schema'] != 'handoff_train_portable_registration_v1'
            or registration['diagnostic_config']['sha256'] != diagnostic_sha256
            or registration['rows'] != 383 or registration['optimizer_updates'] != 0
            or registration['data_split'] != 'train' or output.exists()
            or registration['models'] != ['control','candidate']):
        raise ValueError('Unregistered diagnostic transport')
    bound(repo,registration['diagnostic_config'])
    accepted = read(bound(root,registration['accepted_prefix_audit']))
    check_prefix_rows(cpu_rows,accepted)
    if len(prepared_rows) != 383: raise ValueError('Incomplete prepared rows')
    arch = portable_arch(model_config['arch'],root)
    specs = read(bound(repo,diagnostic['model_specs_only']))['models']
    reserve = registration['reserve_shared_bytes']
    if reserve < (775<<30) or shutil.disk_usage(root).free < reserve+(19<<30):
        raise ValueError('Protect the running champion disk reservation')
    output.mkdir(parents=True); (output/'assets/hf').mkdir(parents=True)
    source_assets = root/'models/qwen3_5_2b_base_processor'
    for source in sorted(source_assets.iterdir()):
        if source.is_symlink() or not source.is_file(): raise ValueError('Unexpected tokenizer asset')
        shutil.copyfile(source,output/'assets/hf'/source.name)
    shutil.copyfile(root/'models/action_tokenizer.pt',output/'assets/action_tokenizer.pt')
    write_new(output/'arch.json',arch)
    # No table reader, video decoder, simulator, optimizer or future frames are
    # needed on the destination. Exact accepted pixels/tokens/targets are retained.
    torch.save(prepared_rows,output/'inputs.pt')
    restored = torch.load(output/'inputs.pt',map_location='cpu',weights_only=False)
    if tree_sha(restored) != tree_sha(prepared_rows): raise ValueError('Portable input bytes changed')
    del restored
    models = {}
    for name in registration['models']:
        spec = specs[name]; path = bound(root,spec['checkpoint'])
        validate_fixed_fit(read(bound(root,spec['fit_result'])),read(bound(root,spec['latest'])),spec['checkpoint'])
        original = torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        state = original['model_state_dict']
        if original['state']['step'] != 360 or len(state) != 950:
            raise ValueError('Wrong full high-planner checkpoint')
        if shutil.disk_usage(root).free < reserve+sum(v.numel()*v.element_size() for v in state.values()):
            raise ValueError('Disk reserve changed before model export')
        target = output/(name+'-model.pt')
        torch.save(dict(model_state_dict=state,state=dict(step=360),source_sha256=spec['checkpoint']['sha256']),target)
        roundtrip = torch.load(target,map_location='cpu',mmap=True,weights_only=False)['model_state_dict']
        if set(roundtrip) != set(state): raise ValueError('Incomplete model-only export')
        for key,value in state.items():
            other = roundtrip[key]
            if (value.dtype != other.dtype or value.shape != other.shape or not torch.equal(
                    value.contiguous().reshape(-1).view(torch.uint8),other.contiguous().reshape(-1).view(torch.uint8))):
                raise ValueError('Model tensor bytes changed: '+key)
        models[name] = dict(original=spec,path=target.name,sha256=file_sha(target),bytes=target.stat().st_size,
                            exact_model_tensors=950,optimizer_state_exported=False)
        del roundtrip,state,original
    write_new(output/'cpu-prefix-audit.json',accepted)
    files = [dict(path=p.relative_to(output).as_posix(),size=p.stat().st_size,sha256=file_sha(p))
             for p in sorted(output.rglob('*')) if p.is_file()]
    manifest = dict(schema='handoff_train_portable_v1',status='exact_inputs_and_models_exported',
        registration_sha256=registration_sha256,diagnostic_sha256=diagnostic_sha256,source_commit=common['source_commit'],
        accepted_prefix_sha256=registration['accepted_prefix_audit']['sha256'],
        accepted_prefix_rows_sha256=digest(cpu_rows),input_tree_sha256=tree_sha(prepared_rows),
        data_split='train',rows=383,optimizer_updates=0,model_forwards=0,models=models,files=files,
        common=common,all_original_targets_unchanged=True,physical_success_claim=False)
    write_new(output/'manifest.json',manifest)
    return dict(path=str(output),manifest_sha256=file_sha(output/'manifest.json'),
                total_bytes=sum(r['size'] for r in files),model_forwards=0,optimizer_updates=0)


def verify_pack(root, manifest_sha256, registration, registration_sha256):
    root = Path(root).resolve(); manifest = read(bound(root,dict(path='manifest.json',sha256=manifest_sha256)))
    if (manifest['schema'] != 'handoff_train_portable_v1' or manifest['status'] != 'exact_inputs_and_models_exported'
            or manifest['registration_sha256'] != registration_sha256
            or manifest['diagnostic_sha256'] != registration['diagnostic_config']['sha256']
            or manifest['data_split'] != 'train' or manifest['rows'] != 383
            or manifest['optimizer_updates'] != 0 or manifest['model_forwards'] != 0
            or not manifest['all_original_targets_unchanged']
            or manifest['accepted_prefix_sha256'] != registration['accepted_prefix_audit']['sha256']):
        raise ValueError('Foreign or incomplete portable TRAIN diagnostic')
    seen = set()
    for row in manifest['files']:
        path = bound(root,dict(path=row['path'],sha256=row['sha256']))
        if row['path'] in seen or (root/row['path']).is_symlink() or path.stat().st_size != row['size']:
            raise ValueError('Duplicate or changed transport artifact')
        seen.add(row['path'])
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root/'manifest.json'}
    if actual != seen or not {'inputs.pt','arch.json','cpu-prefix-audit.json','control-model.pt','candidate-model.pt'} <= seen:
        raise ValueError('Missing or unregistered transport files')
    for name in ('control','candidate'):
        model = manifest['models'][name]
        if (model['exact_model_tensors'] != 950 or model['optimizer_state_exported'] is not False
                or model['path'] != name+'-model.pt'
                or model['sha256'] != next(r['sha256'] for r in manifest['files'] if r['path']==model['path'])):
            raise ValueError('Invalid standalone model binding')
    return manifest
