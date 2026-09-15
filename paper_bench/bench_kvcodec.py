"""Replay completed paper dumps through token-axis FFmpeg HEVC, without fitting."""
import argparse,json,hashlib,time,subprocess,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from safetensors import safe_open
from paper_bench import hevc_token as codec

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,r):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(p)
def objects(rec):
    xs=[]
    for g in rec['groups']:
        if sha(g['path'])!=g['sha256']:raise ValueError('Source cache hash mismatch: '+g['path'])
        with safe_open(g['path'],framework='pt',device='cpu') as f:xs.append(f.get_tensor('x').numpy())
    return [[x[:,i:i+256,:] for x in xs] for i in range(0,rec['slots'],256)]
def restore(packets,pool,ffmpeg,validate=False):
    outputs=[];pending=[]
    def upload(items):
        # Single GPU transfer/dequantization per group of up to eight equal-size objects.
        q=torch.from_numpy(np.stack([x[0] for x in items])).cuda()
        s=torch.from_numpy(np.stack([x[1] for x in items])).cuda()
        y=((q.float()-128)*s[:,:,None,None]).half()
        if validate:outputs.extend(y.cpu().numpy())
    for item in pool.map(lambda p:codec.decode(p,ffmpeg),packets):
        if pending and item[0].shape!=pending[0][0].shape:upload(pending);pending=[]
        pending.append(item)
        if len(pending)==8:upload(pending);pending=[]
    if pending:upload(pending)
    torch.cuda.synchronize()
    return outputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['qwen','glm','deepseek']);p.add_argument('--workers',type=int,default=32);p.add_argument('--repeats',type=int,default=5);p.add_argument('--ffmpeg',default='ffmpeg');a=p.parse_args()
    if a.workers<1 or a.repeats<1:raise ValueError('Positive workers/repeats required')
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=True)
    version=subprocess.check_output([a.ffmpeg,'-version'],text=True)
    encoders=subprocess.check_output([a.ffmpeg,'-hide_banner','-encoders'],text=True,stderr=subprocess.STDOUT)
    if 'libx265' not in encoders:raise ValueError('FFmpeg requires libx265')
    # Real FFmpeg byte-exact smoke checks before the expensive document sweep.
    rng=np.random.default_rng(13)
    for model,groups,d in [('qwen',[4]*16,256),('glm',[4,4,3],512),('deepseek',[3]*7,512)]:
        for length in (1,144,256):
            xs=[rng.normal(size=(v,length,d)).astype(np.float16) for v in groups]
            packet=codec.encode(xs,model,.0267,a.ffmpeg);q,scale=codec.decode(packet,a.ffmpeg);expected,s,_=codec.quantize(xs,.0267)
            if not np.array_equal(q,expected) or not np.array_equal(scale,s):raise ValueError('HEVC geometry selftest failed')
    print('Token layout + FFmpeg exact roundtrip passed for all three geometries and tails',flush=True)
    config=dict(source=str(a.source.resolve()),models=a.models,workers=a.workers,repeats=a.repeats,ffmpeg=version,code={str(p.name):sha(p) for p in (Path(__file__),Path(codec.__file__))})
    cp=a.output/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Changed benchmark configuration; use a new output')
    save(cp,config)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for model in a.models:
            source=a.source/model;manifest=json.loads((source/'manifest.json').read_text());ours=json.loads((source/'result.json').read_text())
            if manifest['status']!='complete' or ours['status']!='complete':raise ValueError('Incomplete source '+model)
            out=a.output/model;out.mkdir(exist_ok=True);rp=out/'result.json'
            report=json.loads(rp.read_text()) if rp.exists() else dict(status='running',model=model,source_manifest_sha256=sha(source/'manifest.json'),ours_result_sha256=sha(source/'result.json'),source_scope=ours['source_scope'],epsilon=ours['epsilon'],rotation=False,centering=False,quantization=getattr(codec,'QUANTIZATION_DESCRIPTION','symmetric INT8; per-view BF16 scales rounded upward; max(sqrt(12)*epsilon*RMS, amax/127); no clipping'),codec=getattr(codec,'CODEC_DESCRIPTION','FFmpeg libx265 lossless, veryfast, token frames, 32 independent processes by default'),timing='CPU FP16 to CPU packets; CPU packets to GPU FP16. Includes FFmpeg process startup, quantization/layout/transfers. Excludes file reads, validation, calibration. No NVDEC or SGLang page writes.',rows=[])
            if report['source_manifest_sha256']!=sha(source/'manifest.json') or report['ours_result_sha256']!=sha(source/'result.json'):raise ValueError('Source report changed')
            for rec in manifest['records']:
                if rec['split']!='b':continue
                todo=[m for m in (.5,.75,1.,1.25) if not any(x['domain']==rec['domain'] and x['multiplier']==m for x in report['rows'])]
                if not todo:continue
                data=objects(rec);rawbytes=sum(x.nbytes for xs in data for x in xs)
                for mult in todo:
                    eps=report['epsilon']*mult
                    print('ENCODE',model,rec['domain'],mult,'objects',len(data),flush=True)
                    packets=list(pool.map(lambda xs:codec.encode(xs,model,eps,a.ffmpeg),data))
                    # Verify the lossless codec against the pre-HEVC codes, outside timing.
                    for i,(xs,(q,scales)) in enumerate(zip(data,pool.map(lambda p:codec.decode(p,a.ffmpeg),packets))):
                        ref,s,_=codec.quantize(xs,eps)
                        if not np.array_equal(q,ref) or not np.array_equal(s,scales):raise ValueError('HEVC changed quantized codes/scales')
                    outputs=restore(packets,pool,a.ffmpeg,True)
                    en=err=f8err=0.;kinds={};maxerr=0.
                    for xs,y in zip(data,outputs):
                        pos=0;oe=os=0.
                        for g,x in zip(rec['groups'],xs):
                            z=y[pos:pos+len(x)];pos+=len(x);x64=x.astype(np.float64)
                            energy=float(np.sum(x64*x64));sse=float(np.sum((z.astype(np.float64)-x64)**2))
                            f8=torch.from_numpy(x.copy()).float().clamp(-448,448).to(torch.float8_e4m3fn).float().numpy()
                            fse=float(np.sum((f8.astype(np.float64)-x64)**2));en+=energy;err+=sse;f8err+=fse;oe+=energy;os+=sse
                            k=kinds.setdefault(g['kind'],dict(energy=0.,sse=0.,fp8_sse=0.));k['energy']+=energy;k['sse']+=sse;k['fp8_sse']+=fse
                        maxerr=max(maxerr,(os/oe)**.5)
                    del outputs
                    heads=[codec.parse(x) for x in packets]
                    row=dict(domain=rec['domain'],task=rec['task'],input_tokens=rec['tokens'],cache_positions=rec['slots'],multiplier=mult,values=rawbytes//2,bytes=sum(map(len,packets)),metadata_bytes=sum(h[3] for h in heads),low4_bytes=sum(h[0].get('low4_bytes',0) for h in heads),hevc_bytes=sum(len(h[2]) for h in heads),range_floor_views=sum(h[0]['range_floor_views'] for h in heads),scale_views=sum(h[0]['views'] for h in heads),energy=en,sse=err,fp8_sse=f8err,nrmse=(err/en)**.5,fp8_nrmse=(f8err/en)**.5,maximum=maxerr,by_kind=kinds,exact_codec_roundtrip=True,geometry=[dict(frames=h[0]['frames'],height=h[0]['height'],width=h[0]['width']) for h in (heads[0],heads[-1])],x265_params=codec.parameters(heads[0][0]['frames'],heads[0][0]['height']))
                    row['bpv']=row['bytes']*8/row['values'];del heads
                    if mult==1.:
                        enc=[];dec=[]
                        for repeat in range(a.repeats+1):
                            torch.cuda.synchronize();start=time.perf_counter()
                            new=list(pool.map(lambda xs:codec.encode(xs,model,eps,a.ffmpeg),data));elapsed=time.perf_counter()-start
                            # Length/content determinism is not needed; exact decoded codes were checked above.
                            if repeat:enc.append(elapsed)
                            del new
                            start=time.perf_counter();restore(packets,pool,a.ffmpeg);elapsed=time.perf_counter()-start
                            if repeat:dec.append(elapsed)
                            print('TIMING',model,rec['domain'],'round',repeat,flush=True)
                        row['timing']=dict(raw_bytes=rawbytes,encode_seconds=enc,decode_seconds=dec,encode_GBs=rawbytes/statistics.median(enc)/1e9,decode_GBs=rawbytes/statistics.median(dec)/1e9)
                    report['rows'].append(row);save(rp,report)
                    print('RESULT',model,rec['domain'],mult,'bpv',row['bpv'],'NRMSE',row['nrmse'],flush=True)
                del data
            report['pooled']=[]
            for mult in (.5,.75,1.,1.25):
                rows=[r for r in report['rows'] if r['multiplier']==mult];en=sum(r['energy'] for r in rows);n=sum(r['values'] for r in rows)
                report['pooled'].append(dict(multiplier=mult,bpv=8*sum(r['bytes'] for r in rows)/n,nrmse=(sum(r['sse'] for r in rows)/en)**.5,fp8_nrmse=(sum(r['fp8_sse'] for r in rows)/en)**.5))
            report['status']='complete';save(rp,report)
    print('Result:',a.output,flush=True)
if __name__=='__main__':main()
