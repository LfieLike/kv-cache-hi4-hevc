"""Fit centered, unit-row GPA independently inside each contiguous four-view group."""
from pathlib import Path
import json
import numpy as np
import torch
from safetensors import safe_open
from group4_gpa_math import fit_generalized_procrustes,orthogonality_max_error
from fresh_qwen_dump import digest,write_json


def checked_manifest(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if m.get('status')!='complete' or m.get('stored_dtype')!='fp16':raise ValueError('Need completed FP16 dump')
    a,b=m['splits']['a'],m['splits']['b']
    if a['context_sha256']==b['context_sha256']:raise ValueError('Fit/test contexts overlap')
    if m['heads']!=4:raise ValueError('This runner expects exactly four heads per layer')
    if [x['layer'] for x in a['layers']]!=m['layers'] or [x['layer'] for x in b['layers']]!=m['layers']:
        raise ValueError('Layer order mismatch')
    for split in (a,b):
        for item in split['layers']:
            if digest(item['path'])!=item['sha256']:raise ValueError('Frozen dump hash changed')
            with safe_open(item['path'],framework='pt',device='cpu') as f:
                for kind in ('key','value'):
                    if f.get_slice(kind).get_shape()!=[4,split['tokens'],m['channels']]:raise ValueError('Dump geometry mismatch')
    return m


def fit_one(x,iterations,tolerance,device,expected_views=4):
    # x: four views x calibration tokens x channel. Only this group is visible.
    if x.ndim!=3 or x.shape[0]!=expected_views or expected_views<2 or not np.isfinite(x).all():raise ValueError('Expected finite [views,T,D] for the requested group')
    mean=x.astype(np.float64).mean(axis=1).astype(np.float32)
    y=torch.from_numpy(x.astype(np.float32)-mean[:,None]).to(device)
    y=y/y.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    cross=torch.einsum('itd,jte->ijde',y,y)
    rot,history=fit_generalized_procrustes(cross,iterations=iterations,tolerance=tolerance)
    err=orthogonality_max_error(rot)
    if not np.isfinite(err) or err>1e-3:raise ValueError(f'Invalid GPA orthogonality {err}')
    return rot.cpu().numpy(),mean,dict(history=history,orthogonality_max_error=err,samples=int(x.shape[1]))


def fit(dump_root,output,kinds=('key','value'),iterations=100,tolerance=1e-6,device='cpu'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m=checked_manifest(dump_root)
    provenance=dict(status='running',dump_manifest_sha256=digest(Path(dump_root)/'manifest.json'),
        calibration=m['splits']['a'],evaluation_context_sha256=m['splits']['b']['context_sha256'],
        group_size=4,view_order='layer-major/head-minor',group_scope='four heads within each individual layer; no cross-layer fit',
        objective='center per-view then normalize each token row; generalized Procrustes cosine objective',
        gauge='first head identity independently per group',iterations=iterations,tolerance=tolerance,device=str(device),artifacts={})
    write_json(output/'fit.json',provenance)
    for kind in kinds:
        rotations=[];means=[];groups=[]
        for group,item in enumerate(m['splits']['a']['layers']):
            with safe_open(item['path'],framework='pt',device='cpu') as f:x=f.get_tensor(kind)
            if x.dtype!=torch.float16:raise ValueError('Dump is not truly FP16')
            rot,mean,stats=fit_one(x.numpy(),iterations,tolerance,device)
            rotations.append(rot);means.append(mean)
            groups.append(dict(group=group,layer=item['layer'],views=list(range(4*group,4*group+4)),heads=[0,1,2,3],**stats))
            print(f"fit {kind} layer={item['layer']} group={group}: cosine {stats['history'][0]['mean_pairwise_cosine']:.5f} -> {stats['history'][-1]['mean_pairwise_cosine']:.5f}",flush=True)
        path=output/f'{kind}_group4.npz'
        np.savez(path,rotations=np.stack(rotations),means=np.stack(means),layers=m['layers'],heads=np.arange(4))
        provenance['artifacts'][kind]=dict(path=str(path.resolve()),sha256=digest(path),groups=groups)
        write_json(output/'fit.json',provenance)
    provenance['status']='complete';write_json(output/'fit.json',provenance)


def export(dump_root,fit_root,output,kinds=('key','value'),start=0):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m=checked_manifest(dump_root);fit_meta=json.loads((Path(fit_root)/'fit.json').read_text())
    if fit_meta['status']!='complete' or fit_meta['dump_manifest_sha256']!=digest(Path(dump_root)/'manifest.json'):
        raise ValueError('Fit provenance does not match frozen dump')
    if start<0 or start+1024>m['splits']['b']['tokens']:raise ValueError('Need four consecutive 256-token test blocks')
    for kind in kinds:
        folder=output/kind;folder.mkdir()
        artifact=fit_meta['artifacts'][kind]
        if digest(artifact['path'])!=artifact['sha256']:raise ValueError('Fit artifact changed')
        with np.load(artifact['path']) as f:rot=f['rotations'];mean=f['means']
        items=[]
        for offset in range(start,start+1024,256):
            layers=[]
            for item in m['splits']['b']['layers']:
                with safe_open(item['path'],framework='pt',device='cpu') as f:
                    x=f.get_slice(kind)[:,offset:offset+256,:]
                if x.dtype!=torch.float16:raise ValueError('Expected FP16 test source')
                layers.append(x.permute(1,0,2).numpy())
            path=folder/f'{kind}-seq0-base0-token{offset}.npz'
            np.savez(path,**{kind:np.stack(layers)},rotations=rot,means=mean,source_dtype='fp16',
                layer_indices=m['layers'],head_indices=np.arange(4))
            items.append(dict(path=str(path.resolve()),sha256=digest(path),token_start=offset,token_count=256))
        write_json(folder/'environment.json',dict(status='complete',kind=kind,inputs={'qwen':items},
            fit_artifact=artifact['path'],fit_sha256=artifact['sha256'],
            fit_manifest_sha256=digest(Path(fit_root)/'fit.json'),dump_manifest_sha256=digest(Path(dump_root)/'manifest.json'),
            evaluation_context_sha256=m['splits']['b']['context_sha256'],calibration_context_sha256=m['splits']['a']['context_sha256']))


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['fit','export']);p.add_argument('--dump-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--fit-root',type=Path)
    p.add_argument('--kinds',nargs='+',choices=['key','value'],default=['key','value'])
    p.add_argument('--iterations',type=int,default=100);p.add_argument('--tolerance',type=float,default=1e-6)
    p.add_argument('--fit-threads',type=int,default=4);p.add_argument('--device',default='auto');p.add_argument('--eval-start',type=int,default=0)
    a=p.parse_args()
    if min(a.iterations,a.tolerance,a.fit_threads)<=0:raise ValueError('Invalid fit parameters')
    torch.set_num_threads(a.fit_threads)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    if a.stage=='fit':fit(a.dump_root,a.output,a.kinds,a.iterations,a.tolerance,device)
    else:
        if a.fit_root is None:raise ValueError('--fit-root required')
        export(a.dump_root,a.fit_root,a.output,a.kinds,a.eval_start)

if __name__=='__main__':main()
