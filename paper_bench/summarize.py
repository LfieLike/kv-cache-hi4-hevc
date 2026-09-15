import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();rows=[];summary=[]
for model in ('qwen','glm','deepseek'):
 path=a.root/model/'result.json'
 if not path.exists():continue
 r=json.loads(path.read_text())
 if r['status']!='complete':raise ValueError('Incomplete model '+model)
 manifest_path=a.root/model/'manifest.json'
 records={x['domain']:x for x in json.loads(manifest_path.read_text())['records'] if x['split']=='b'} if manifest_path.exists() else {}
 for x in r['rows']:
  rec=records.get(x['domain'],{})
  rows.append(dict(model=model,domain=x['domain'],input_tokens=x.get('input_tokens',rec.get('tokens','')),cache_positions=x.get('cache_positions',rec.get('slots','')),task=x.get('task',rec.get('task','')),tail_bytes=x.get('tail_bytes',0),multiplier=x['multiplier'],target=r['epsilon']*x['multiplier'],nrmse=x['nrmse'],fp8_nrmse=x['fp8_nrmse'],bpv=x['bpv'],encode_GBs=x.get('timing',{}).get('encode_GBs',''),decode_GBs=x.get('timing',{}).get('decode_GBs','')))
 for point in r['pooled']:
  s=dict(model=model,**point)
  ts=[x['timing'] for x in r['rows'] if x['multiplier']==point['multiplier'] and 'timing' in x]
  if ts:
   n=sum(t['raw_bytes'] for t in ts)
   for op in ('encode','decode'):s[op+'_GBs']=n/float(np.median(np.sum([t[op+'_seconds'] for t in ts],axis=0)))/1e9
  summary.append(s)
if not rows:raise ValueError('No results')
with (a.root/'per_domain.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(a.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(9,2.7),squeeze=False)
for ax,model in zip(axes[0],('qwen','glm','deepseek')):
 points=sorted([x for x in summary if x['model']==model],key=lambda x:x['nrmse'])
 if not points:ax.set_visible(False);continue
 ax.plot([100*x['nrmse'] for x in points],[x['bpv'] for x in points],'o-',color='#ad2851')
 ax.axvline(100*points[0]['fp8_nrmse'],color='#71808b',linestyle='--',linewidth=1,label='FP8 reference')
 ax.set(title=model,xlabel='Measured relative error (%)',ylabel='Bits per value');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(a.root/'ours_rd.pdf');fig.savefig(a.root/'ours_rd.png',dpi=200)
print(json.dumps(summary,indent=2))
