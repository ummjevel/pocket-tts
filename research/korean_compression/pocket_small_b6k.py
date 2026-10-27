"""Portable Korean compression research; native learning algorithms stay upstream."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from project import ROOT, SETTINGS, materialize, checked, guard, scorer, sha
SOURCE = ROOT
OUT = ROOT / SETTINGS['backbone_dir']
STAGES = [('width768',768,2.,12),('width512',512,1.,8)]
RUN_ID = os.environ.get('SLURM_JOB_ID','local')

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def dump(path, data):
    import yaml
    Path(path).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def call(*args):
    subprocess.run([sys.executable, '-Bu', str(Path(__file__).resolve()), *map(str,args)], check=True)


def seed_stage(name):
    import torch
    from training.scripts.residual_cov import residual_cov
    from training.scripts.shrink_checkpoint import flow_lm_shapes, shrink_width
    import yaml
    index = [x[0] for x in STAGES].index(name)
    source_ckpt = ROOT / SETTINGS['assets_dir'] / 'teacher.pt' if index == 0 else OUT / STAGES[index-1][0] / 'checkpoint_00020000.pt'
    source_args = OUT / 'teacher_args.yaml' if index == 0 else OUT / f'{STAGES[index-1][0]}.yaml'
    target = OUT / f'{name}_pca_init.pt'
    if target.exists():
        return
    device = guard()
    cov = residual_cov(str(source_args),str(source_ckpt),32,True)
    torch.save(cov,OUT / f'{name}_cov.pt')
    payload = torch.load(source_ckpt,map_location='cpu',weights_only=True)
    state = dict(payload['model']);state.update(payload.get('ema') or {})
    src = {k.removeprefix('flow_lm.'):v for k,v in state.items() if k.startswith('flow_lm.')}
    cfg = yaml.safe_load((OUT / f'{name}.yaml').read_text())
    dst, head = flow_lm_shapes(cfg['model_config'],{})
    seeded = shrink_width(src,dst,head,cov['cov'])
    assert set(seeded) == set(dst), sorted(set(dst)-set(seeded))
    torch.save({'step':0,'model':{'flow_lm.'+k:v for k,v in seeded.items()},'ema':None},target)
    write(OUT / f'{name}_seed.json',{'source':str(source_ckpt),'source_sha':sha(source_ckpt),
        'covariance_tokens':cov['tokens'],'method':'Unmodified official residual_cov + shrink_width(cov)',
        'complete_flow_seed':True,'parameters':sum(v.numel() for v in seeded.values()),'device':str(device)})


def train(name, smoke=False):
    import torch
    import yaml
    from training import train as T
    from training.args import load_args
    from training.checkpointing import latest_checkpoint
    guard()
    path = OUT / f'{name}.yaml'
    if smoke:
        cfg = yaml.safe_load(path.read_text());cfg.update(run_dir=str(OUT / f'smoke_{name}'),max_steps=3,
            valid_freq=1000000,ckpt_freq=1000000)
        path = OUT / f'smoke_{name}.yaml';dump(path,cfg)
    else:
        run = OUT / name
        own = latest_checkpoint(run)
        if own: assert own.with_name(own.name.replace('checkpoint_','optim_')).is_file()
        if own and own.name == 'checkpoint_00020000.pt': return
    original = T.setup
    def setup(*a,**kw):
        r = original(*a,**kw)
        assert not any(p.requires_grad for p in r.model.distill_teacher.parameters())
        assert not any(p.requires_grad for p in r.model.flow_lm.flow_net.parameters())
        assert not any(p.requires_grad for p in r.model.flow_lm.out_eos.parameters())
        assert r.model.flow_lm.z_proj.weight.requires_grad
        for k,v in r.model.flow_lm.state_dict().items():
            if k.startswith(('flow_net.','out_eos.')):
                assert torch.equal(v.detach().cpu(),r.model.distill_teacher.state_dict()[k].detach().cpu())
        write(OUT / f'{name}_launch_{RUN_ID}_{"smoke" if smoke else "train"}.json',
            {'step':r.start_step,'gpu':str(torch.cuda.get_device_properties(0).uuid),
             'flow_parameters':sum(p.numel() for p in r.model.flow_lm.parameters()),
             'full_bundle_parameters':sum(p.numel() for p in r.model.flow_lm.parameters())+sum(p.numel() for p in r.mimi.parameters()),
             'frozen_heads_verified':True,'global_batch':r.args.batch_size*r.args.grad_accum_steps})
        return r
    T.setup = setup
    origlog = T.ProgressLog.log
    def log(self,event,step,*a,**kw):
        origlog(self,event,step,*a,**kw)
        if not smoke and event=='checkpoint' and step in (2000,6000,10000,20000):
            # Separate subprocesses free scorer/model memory on exit; training is paused.
            call('evaluate',name,'--step',step)
    T.ProgressLog.log = log
    T.main(str(path))
    if smoke:
        latest = OUT / f'smoke_{name}' / 'checkpoint_00000003.pt'
        p = torch.load(latest,map_location='cpu',weights_only=True)
        assert p['step']==3 and p['ema']
        assert all(torch.isfinite(v).all() for v in p['model'].values())
        write(OUT / f'{name}_smoke_passed.json',{'steps':3,'finite_weights':True,'checkpoint':str(latest)})


def evaluate(name, step, stage):
    import torch
    import functools
    from training.eval.librispeech import load_run
    device=guard(); C=scorer()
    target=OUT / 'evaluations' / f'{name}_s{step:08d}';target.mkdir(parents=True,exist_ok=True)
    run=OUT / ('teacher_run' if name=='control' else name)
    ckpt=ROOT / SETTINGS['assets_dir'] / 'teacher.pt' if name=='control' else run / f'checkpoint_{step:08d}.pt'
    use_ema = name != 'control'  # control already contains merged EMA; it has no EMA shadow
    config=C.EvalConfig(C.DATASET_REPO,C.DATASET_REVISION,C.DATASET_FILE,C.DATASET_SPLIT,
        C.ASR_REPO,C.ASR_REVISION,'ko',str(ckpt),step,use_ema,500,0,.3,1.,1,-1.,30.,0.)
    items=C.load_items(Path(SETTINGS['evaluation_data']))
    assert len(items)==500
    C._write_config(target/'config.json',config)
    if stage=='generate':
        model,mimi,actual=load_run(run,device,use_ema,ckpt);assert actual==step
        start=time.monotonic()
        generation=C.generate_items(model,mimi,items,target,device,batch_size=8,seed=0,temp=.3,
            cfg=1.,n_steps=1,eos_threshold=-1.,max_sec=30.,voice_sec=0.)
        write(target/'generation_time.json',{'seconds':time.monotonic()-start,'batch_size':8,
            'note':'Evaluation batch throughput, not batch1 deployment RTF'})
        assert len(generation)==500
    else:
        generation=C._jsonl_map(target/'generation.jsonl');assert len(generation)==500
        metrics=Path(SETTINGS['metrics_dir'])
        if stage=='asr':
            C.ASR_REPO=str(metrics/'whisper-large-v3')
            records=C.transcribe_items(items,generation,target,device,16);assert len(records)==500
        elif stage=='quality':
            import utmos_pytorch
            utmos_pytorch.UTMOSScoreTorch=functools.partial(utmos_pytorch.UTMOSScoreTorch,
                ckpt_path=str(metrics/'utmos-v1/utmos_state_dict.pt'))
            C.SPEAKER_REPO=str(metrics/'wavlm-base-plus-sv')
            records=C.score_quality_items(items,generation,target,device,16,0.);assert len(records)==500
        elif stage=='report':
            asr=C._jsonl_map(target/'asr.jsonl');quality=C._jsonl_map(target/'quality.jsonl')
            assert set(asr)==set(quality)==set(generation)
            write(target/'results.json',C.summarize(items,asr,config,quality))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seed','train','smoke','evaluate','evalstage','pipeline'])
    p.add_argument('name',nargs='?');p.add_argument('--step',type=int,default=6000);p.add_argument('--stage')
    a=p.parse_args();materialize();checked();sys.path.insert(0,str(SOURCE))
    if a.action=='seed':seed_stage(a.name)
    elif a.action=='train':train(a.name)
    elif a.action=='smoke':train(a.name,True)
    elif a.action=='evalstage':evaluate(a.name,a.step,a.stage)
    elif a.action=='evaluate':
        if a.name!='control' and not (OUT/'evaluations/control_s00006000/results.json').exists():call('evaluate','control')
        for s in ('generate','asr','quality','report'):call('evalstage',a.name,'--step',a.step,'--stage',s)
    else:
        if not (OUT/'evaluations/control_s00006000/results.json').exists():call('evaluate','control')
        for name,*_ in STAGES:
            call('seed',name)
            if not (OUT/f'{name}_smoke_passed.json').exists():call('smoke',name)
            call('train',name)
        write(OUT/'pipeline_done.json',{'phase':'backbone only'})
if __name__=='__main__':main()
