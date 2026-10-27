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
OUT = ROOT / SETTINGS['followup_dir']
STAGES = [('width768',768,2.,12),('width512',512,1.,8)]
RUN_ID = os.environ.get('SLURM_JOB_ID','local')

import pocket_small_b6k as P
PHASE=P.OUT

def dump(path,data):
    import yaml
    Path(path).write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False))


def yamlread(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def call(*args):
    subprocess.run([sys.executable,'-Bu',str(Path(__file__).resolve()),*map(str,args)],check=True)


def seed224():
    import torch
    from training.scripts.residual_cov import residual_cov
    from training.scripts.shrink_checkpoint import flow_lm_shapes,shrink_width
    P.guard()
    target=OUT/'width224_pca_init.pt'
    if target.exists(): return
    ckpt=PHASE/'width512/checkpoint_00020000.pt'
    cov=residual_cov(str(PHASE/'width512.yaml'),str(ckpt),32,True)
    torch.save(cov,OUT/'width224_cov.pt')
    raw=torch.load(ckpt,map_location='cpu',weights_only=True)
    state={**raw['model'],**raw['ema']}
    src={k.removeprefix('flow_lm.'):v for k,v in state.items() if k.startswith('flow_lm.')}
    dst,h=flow_lm_shapes(str(OUT/'width224_model.yaml'),{})
    seeded=shrink_width(src,dst,h,cov['cov']);assert set(seeded)==set(dst),set(dst)-set(seeded)
    torch.save({'step':0,'model':{'flow_lm.'+k:v for k,v in seeded.items()},'ema':None},target)


def train(name,smoke):
    import torch
    from training import train as T
    from training.checkpointing import latest_checkpoint
    P.guard();config=OUT/f'{name}.yaml';cfg=yamlread(config)
    if smoke:
        cfg.update(run_dir=str(OUT/f'smoke_{name}'),max_steps=3,ckpt_freq=1000000,valid_freq=1000000)
        config=OUT/f'smoke_{name}.yaml';dump(config,cfg)
    else:
        own=latest_checkpoint(Path(cfg['run_dir']))
        if own:
            assert own.with_name(own.name.replace('checkpoint_','optim_')).is_file()
            if int(own.stem.split('_')[-1])==cfg['max_steps']: return
    old=T.setup
    def setup(*a,**kw):
        r=old(*a,**kw)
        if name.startswith('head'):
            assert all(not p.requires_grad for n,p in r.model.flow_lm.named_parameters() if not n.startswith('flow_net.'))
            assert all(p.requires_grad for p in r.model.flow_lm.flow_net.parameters())
            assert not any(p.requires_grad for p in r.model.head_teacher.parameters())
        else:
            assert not any(p.requires_grad for p in r.model.flow_lm.flow_net.parameters())
        P.write(OUT/f'{name}_{"smoke" if smoke else "train"}_launch.json',{'job':RUN_ID,
            'step':r.start_step,'flow_parameters':sum(p.numel() for p in r.model.flow_lm.parameters()),
            'heads_only':name.startswith('head'),'gpu':str(torch.cuda.get_device_properties(0).uuid)})
        return r
    T.setup=setup
    oldlog=T.ProgressLog.log
    def log(self,event,step,*a,**kw):
        oldlog(self,event,step,*a,**kw)
        if not smoke and event=='checkpoint' and step in (2000,6000,10000,20000):
            call('evaluate',name,'--step',step,'--codec','full')
    T.ProgressLog.log=log
    T.main(str(config))
    if smoke:
        payload=torch.load(Path(cfg['run_dir'])/'checkpoint_00000003.pt',map_location='cpu',weights_only=True)
        assert payload['step']==3 and payload['ema']
        assert all(torch.isfinite(v).all() for v in payload['model'].values())
        P.write(OUT/f'{name}_smoke_passed.json',{'finite_weights':True,'steps':3})


def make_mini_run(name):
    cfg=yamlread(OUT/f'{name}.yaml');model=yamlread(cfg['model_config'])
    model['mimi']=yamlread(SOURCE/'pocket_tts/config/english_18m_2026-10.yaml')['mimi']
    model['weights_path']=str(OUT/'hybrid_small_codec.safetensors')
    model['weights_path_without_voice_cloning']=model['weights_path']
    dump(OUT/f'{name}_mini_model.yaml',model)
    cfg['model_config']=str(OUT/f'{name}_mini_model.yaml')
    folder=OUT/f'{name}_mini';folder.mkdir(exist_ok=True);dump(folder/'args.yaml',cfg)
    return folder


def evaluate(name,step,codec,stage):
    import torch
    import functools
    import hashlib
    from training.eval.librispeech import load_run
    device=P.guard();C=P.scorer()
    target=OUT/'evaluations'/f'{name}_{codec}_s{step:08d}';target.mkdir(parents=True,exist_ok=True)
    run=OUT/name if codec=='full' else make_mini_run(name)
    checkpoint=OUT/name/f'checkpoint_{step:08d}.pt'
    config=C.EvalConfig(C.DATASET_REPO,C.DATASET_REVISION,C.DATASET_FILE,C.DATASET_SPLIT,C.ASR_REPO,
        C.ASR_REVISION,'ko',str(checkpoint),step,True,500,0,.3,1.,1,-1.,30.,0.)
    items=C.load_items(Path(SETTINGS['evaluation_data']))
    C._write_config(target/'config.json',config)
    if stage=='generate':
        if (target/'latent_hashes.json').exists(): return
        assert not (target/'generation.jsonl').exists(),'Interrupted generation needs a new attempt to keep complete latent hashes'
        model,mimi,actual=load_run(run,device,True,checkpoint);assert actual==step
        hashes=[];original=model.generate
        def generate(*a,**kw):
            latents=original(*a,**kw)
            hashes.extend(hashlib.sha256(x.detach().float().cpu().numpy().tobytes()).hexdigest() for x in latents)
            return latents
        model.generate=generate
        rows=C.generate_items(model,mimi,items,target,device,batch_size=8,seed=0,temp=.3,cfg=1.,
            n_steps=1,eos_threshold=-1.,max_sec=30.,voice_sec=0.)
        assert len(rows)==len(hashes)==500
        P.write(target/'latent_hashes.json',dict(zip((i.id for i in items),hashes)))
        return
    generation=C._jsonl_map(target/'generation.jsonl');assert len(generation)==500
    metrics=Path(SETTINGS['metrics_dir'])
    if stage=='asr':
        C.ASR_REPO=str(metrics/'whisper-large-v3');assert len(C.transcribe_items(items,generation,target,device,16))==500
    elif stage=='quality':
        import utmos_pytorch
        utmos_pytorch.UTMOSScoreTorch=functools.partial(utmos_pytorch.UTMOSScoreTorch,
            ckpt_path=str(metrics/'utmos-v1/utmos_state_dict.pt'))
        C.SPEAKER_REPO=str(metrics/'wavlm-base-plus-sv')
        assert len(C.score_quality_items(items,generation,target,device,16,0.))==500
    else:
        asr=C._jsonl_map(target/'asr.jsonl');quality=C._jsonl_map(target/'quality.jsonl')
        assert set(asr)==set(quality)==set(generation)
        P.write(target/'results.json',C.summarize(items,asr,config,quality))


def decoder_gate(name):
    def folder(c): return OUT/'evaluations'/f'{name}_{c}_s00010000'
    full=json.loads((folder('full')/'results.json').read_text())
    mini=json.loads((folder('mini')/'results.json').read_text())
    same=json.loads((folder('full')/'latent_hashes.json').read_text())==json.loads((folder('mini')/'latent_hashes.json').read_text())
    tests={'identical_latents':same,'cer':mini['cer']<=full['cer']+.01,
        'utmos':full['utmos'] is not None and mini['utmos'] is not None and mini['utmos']>=full['utmos']-.05,
        'speaker_similarity':full['speaker_similarity'] is not None and mini['speaker_similarity'] is not None and mini['speaker_similarity']>=full['speaker_similarity']-.02,
        'silent':mini['silent']==0,'no_eos':mini['no_eos']<=full['no_eos']}
    P.write(OUT/f'{name}_decoder_gate.json',{'passed':all(tests.values()),'tests':tests,
        'baseline':full,'small_decoder':mini,'policy':'Failed gate retains original decoder; no automatic quality claim or deployment.'})

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pipeline','seed224','train','smoke','evaluate','evalstage','gate'])
    p.add_argument('name',nargs='?');p.add_argument('--step',type=int,default=10000)
    p.add_argument('--codec',choices=['full','mini'],default='full');p.add_argument('--stage')
    a=p.parse_args()
    materialize();checked();sys.path.insert(0,str(SOURCE))
    if a.action=='seed224':seed224()
    elif a.action=='train':train(a.name,False)
    elif a.action=='smoke':train(a.name,True)
    elif a.action=='evaluate':
        for s in ('generate','asr','quality','report'):call('evalstage',a.name,'--step',a.step,'--codec',a.codec,'--stage',s)
    elif a.action=='evalstage':evaluate(a.name,a.step,a.codec,a.stage)
    elif a.action=='gate':decoder_gate(a.name)
    elif a.action=='pipeline':
        assert (PHASE/'pipeline_done.json').is_file(), 'Backbone pipeline must finish successfully'
        control=PHASE/'evaluations/control_s00006000/results.json'
        if not control.exists():P.call('evaluate','control')
        c=json.loads(control.read_text());assert c['silent']==0 and c['cer']<.10,'Invalid B6k control; stop'
        for head in ('head512','head224'):
            if head=='head224':
                call('seed224')
                if not (OUT/'width224_smoke_passed.json').exists():call('smoke','width224')
                call('train','width224')
            if not (OUT/f'{head}_smoke_passed.json').exists():call('smoke',head)
            call('train',head)
            call('evaluate',head,'--codec','mini')
            call('gate',head)
        P.write(OUT/'pipeline_done.json',{'time':time.time(),'scope':'Small backbone+small native drifting head+decoder A/B gates; not exact release/z64 reproduction'})

if __name__=='__main__':main()
