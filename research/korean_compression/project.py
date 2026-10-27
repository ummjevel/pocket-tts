"""Local-only environment boundary for Korean compression experiments."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
CONFIG=Path(os.environ.get('POCKET_LOCAL_CONFIG',str(ROOT/'.local/settings.json')))
SETTINGS=json.loads((CONFIG if CONFIG.exists() else ROOT/'configs/korean_compression/settings.example.json').read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def location(path):
    p=Path(path).expanduser()
    return p if p.is_absolute() else ROOT/p


def materialize():
    import yaml
    if not CONFIG.exists():
        raise RuntimeError('Copy configs/korean_compression/settings.example.json to .local/settings.json first')
    os.chdir(ROOT)
    for key in ('evaluation_data','metrics_dir','train_manifest','valid_manifest'):
        SETTINGS[key]=str(location(SETTINGS[key]))
    values={'BACKBONE':str(location(SETTINGS['backbone_dir'])),
        'FOLLOWUP':str(location(SETTINGS['followup_dir'])),
        'ASSETS':str(location(SETTINGS['assets_dir'])),
        'TRAIN_MANIFEST':SETTINGS['train_manifest'],'VALID_MANIFEST':SETTINGS['valid_manifest']}
    def resolve(v):
        if isinstance(v,str):
            for key,path in values.items():v=v.replace('${'+key+'}',path)
            return v
        if isinstance(v,list):return [resolve(x) for x in v]
        if isinstance(v,dict):return {k:resolve(x) for k,x in v.items()}
        return v
    for kind,root in [('backbone',values['BACKBONE']),('followup',values['FOLLOWUP'])]:
        target=Path(root);target.mkdir(parents=True,exist_ok=True)
        for p in (ROOT/'configs/korean_compression'/kind).glob('*.yaml'):
            d=resolve(yaml.safe_load(p.read_text()))
            if 'batch_size' in d:
                d['batch_size']=SETTINGS['microbatch'];d['grad_accum_steps']=SETTINGS['accumulation']
                d['data']['loader_procs']=SETTINGS.get('loader_procs',1)
                assert d['batch_size']*d['grad_accum_steps']==64
            (target/p.name).write_text(yaml.safe_dump(d,allow_unicode=True,sort_keys=False))
    teacher=Path(values['BACKBONE'])/'teacher_run';teacher.mkdir(exist_ok=True)
    (teacher/'args.yaml').write_text((teacher.parent/'teacher_args.yaml').read_text())


def checked():
    path=ROOT/'.local/artifacts_manifest.json'
    if not path.exists():raise RuntimeError('Transfer the ignored artifacts pack and its hash manifest first')
    data=json.loads(path.read_text())
    for name,record in data['files'].items():
        p=location(SETTINGS['assets_dir'])/name
        with p.open('rb') as f:
            h=hashlib.sha256()
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==record['sha256'],f'Artifact integrity failure: {name}'
    return data


def guard():
    import torch
    device=torch.device(SETTINGS.get('device','cpu'))
    if device.type=='cuda':
        if SETTINGS.get('require_slurm',False) and not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('This local profile requires a real scheduler allocation')
        assert torch.cuda.is_available(),'CUDA unavailable; check the installed torch build'
        torch.cuda.set_device(device)
        allowed=SETTINGS.get('allowed_gpu_uuids',[])
        if allowed:
            uuid=str(torch.cuda.get_device_properties(device).uuid).removeprefix('GPU-')
            assert uuid in allowed,'GPU does not match the local allowlist'
        torch.cuda.set_per_process_memory_fraction(SETTINGS.get('memory_fraction',.85),device)
    torch.set_num_threads(SETTINGS.get('threads',4))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return device


def scorer():
    sys.path.insert(0,str(ROOT))
    path=Path(__file__).with_name('cv3_ko.py')
    spec=importlib.util.spec_from_file_location('korean_cv3_scorer',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module
