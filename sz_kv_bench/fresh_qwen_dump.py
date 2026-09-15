"""Fresh Qwen full-attention cache collection; adapted from collect_qwen35_kv.py.
No old dump reuse. Stores FP16 K/V, records model compute dtype separately.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any
import numpy as np
import torch
from safetensors.torch import save_file
NUM_KV_HEADS=4
HEAD_DIM=256
LAYERS=list(range(3,32,4))
BOOKS={
 'a':('longbook_qa_eng.jsonl','660892d133a7669198428ad0c92d6ce8404312af742b728391a1cdf9524fab93'),
 'b':('longbook_sum_eng.jsonl','3928828a83a42d47d1e4f93bc28d8bdea7cd8282814505fa0dd6dfda72d4802a'),
}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def read_book(root,split):
    name,want=BOOKS[split]
    paths=sorted(Path(root).rglob(name))
    if not paths:raise FileNotFoundError(f'{name} not found under {root}')
    for path in paths:
        with path.open() as f:
            for line_no,line in enumerate(f,1):
                row=json.loads(line);text=row.get('context')
                if isinstance(text,str) and hashlib.sha256(text.encode()).hexdigest()==want:
                    return text,dict(path=str(path.resolve()),line=line_no,context_sha256=want)
    raise ValueError(f'Official Split-{split} book SHA256 not found')

def _as_htd(key: torch.Tensor, value: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a layer cache to ``[H,T,D]`` with H=4, D=256."""

    if key is None or value is None:
        raise TypeError(f"layer {layer} cache is empty")
    if key.shape != value.shape:
        raise ValueError(f"layer {layer} key {tuple(key.shape)} != value {tuple(value.shape)}")
    work_k, work_v = key, value
    if work_k.ndim == 5 and int(work_k.shape[0]) == 1:
        work_k, work_v = work_k[0], work_v[0]
    if work_k.ndim == 4:
        if int(work_k.shape[0]) == 1:
            work_k, work_v = work_k[0], work_v[0]
        elif int(work_k.shape[1]) == NUM_KV_HEADS and int(work_k.shape[-1]) == HEAD_DIM:
            work_k, work_v = work_k.permute(1, 0, 2).contiguous(), work_v.permute(1, 0, 2).contiguous()
        else:
            raise ValueError(f"layer {layer} key {tuple(key.shape)} is not [B,H,T,D] or [B,T,H,D]")
    if work_k.ndim == 3 and int(work_k.shape[1]) == NUM_KV_HEADS and int(work_k.shape[-1]) == HEAD_DIM:
        if int(work_k.shape[0]) != NUM_KV_HEADS:
            work_k, work_v = work_k.permute(1, 0, 2).contiguous(), work_v.permute(1, 0, 2).contiguous()
    if work_k.ndim != 3 or int(work_k.shape[0]) != NUM_KV_HEADS or int(work_k.shape[-1]) != HEAD_DIM:
        raise ValueError(f"layer {layer} key {tuple(key.shape)} is not [H,T,D]")
    return work_k.detach().float().contiguous(), work_v.detach().float().contiguous()

def _pair_from_layer_obj(entry: Any) -> tuple[Any, Any]:
    if hasattr(entry, "keys") and hasattr(entry, "values"):
        key = entry.keys
        value = entry.values
        if callable(key) or callable(value):
            key = getattr(entry, "key_cache", None)
            value = getattr(entry, "value_cache", None)
        return key, value
    if hasattr(entry, "key_cache") and hasattr(entry, "value_cache"):
        return entry.key_cache, entry.value_cache
    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        return entry[0], entry[1]
    return None, None

def _extract_layer_kv(past: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one GQA layer from HF Cache / DynamicCache / legacy tuple."""

    key = value = None
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        kc, vc = past.key_cache, past.value_cache
        if isinstance(kc, (list, tuple)) and layer < len(kc):
            key, value = kc[layer], vc[layer]
    if key is None and hasattr(past, "layers"):
        layers = past.layers
        if isinstance(layers, (list, tuple)) and layer < len(layers):
            key, value = _pair_from_layer_obj(layers[layer])
    if key is None and hasattr(past, "to_legacy_cache"):
        legacy = past.to_legacy_cache()
        if isinstance(legacy, (list, tuple)) and layer < len(legacy):
            key, value = _pair_from_layer_obj(legacy[layer])
    if key is None and isinstance(past, (tuple, list)):
        key, value = _pair_from_layer_obj(past[layer])
    if key is None:
        raise TypeError(
            f"cannot read K/V from {type(past).__name__} layer={layer}; "
            "need DynamicCache.key_cache, Cache.layers, or a legacy tuple"
        )
    return _as_htd(key, value, layer)

def load_model(model_dir: Path, *, device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    if device not in ("auto", "cpu") and not hasattr(model, "hf_device_map"):
        model = model.to(device)
    model.eval()
    return tokenizer, model

def prefill(model,ids,chunk):
    params=set(inspect.signature(model.forward).parameters)
    past=None
    for start in range(0,ids.shape[1],chunk):
        end=min(start+chunk,ids.shape[1]);piece=ids[:,start:end]
        kwargs=dict(input_ids=piece,use_cache=True)
        if 'attention_mask' in params:kwargs['attention_mask']=torch.ones((1,end),dtype=torch.long,device=piece.device)
        if 'position_ids' in params:kwargs['position_ids']=torch.arange(start,end,device=piece.device)[None]
        if 'logits_to_keep' in params:kwargs['logits_to_keep']=1
        if past is not None:kwargs['past_key_values']=past
        started=time.perf_counter()
        with torch.inference_mode():out=model(**kwargs)
        past=getattr(out,'past_key_values',None)
        if past is None:past=getattr(out,'cache',None)
        if past is None:raise RuntimeError('Model returned no KV cache')
        del out
        print(f'prefill {end}/{ids.shape[1]}: {time.perf_counter()-started:.2f}s',flush=True)
    return past


def collect(output,model_path,data_root,fit_tokens,eval_tokens,chunk,device,dtype):
    import transformers
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    texts={};sources={}
    for split in ('a','b'):texts[split],sources[split]=read_book(data_root,split)
    if sources['a']['context_sha256']==sources['b']['context_sha256']:raise ValueError('Calibration/evaluation overlap')
    config=json.loads((Path(model_path)/'config.json').read_text())
    text_config=config.get('text_config',config)
    if int(text_config.get('num_key_value_heads',4))!=4:raise ValueError('Expected Qwen four KV heads')
    layer_types=text_config.get('layer_types')
    if layer_types and [i for i,t in enumerate(layer_types) if t=='full_attention']!=LAYERS:
        raise ValueError('Model full-attention layer geometry differs from expected Qwen3.5-9B')
    torch.manual_seed(0)
    tokenizer,model=load_model(Path(model_path),device=device,dtype=dtype)
    manifest=dict(status='running',model=str(Path(model_path).resolve()),model_config_sha256=digest(Path(model_path)/'config.json'),
        model_compute_dtype=dtype,stored_dtype='fp16',seed=0,prefill_chunk=chunk,
        model_files=[dict(name=p.name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(Path(model_path).glob('*.safetensors'))],
        tokenizer_files={p.name:digest(p) for p in sorted(Path(model_path).glob('*token*')) if p.is_file()},torch_version=torch.__version__,transformers_version=transformers.__version__,
        layers=LAYERS,heads=NUM_KV_HEADS,channels=HEAD_DIM,sources=sources,splits={})
    write_json(output/'manifest.json',manifest)
    for split,tokens in (('a',fit_tokens),('b',eval_tokens)):
        ids=tokenizer(texts[split],add_special_tokens=False,return_tensors='pt')['input_ids']
        if ids.shape[1]<tokens:raise ValueError(f'Split-{split} shorter than {tokens} tokens')
        ids=ids[:,:tokens].contiguous()
        folder=output/split;folder.mkdir()
        token_path=folder/'input_ids.npy';np.save(token_path,ids.numpy())
        ids=ids.to(model.get_input_embeddings().weight.device)
        past=prefill(model,ids,chunk);layers=[]
        for layer in LAYERS:
            k,v=_extract_layer_kv(past,layer)
            if tuple(k.shape)!=(NUM_KV_HEADS,tokens,HEAD_DIM):raise ValueError(f'Unexpected cache shape {tuple(k.shape)}')
            k=k.cpu().to(torch.float16);v=v.cpu().to(torch.float16)
            if not torch.isfinite(k).all() or not torch.isfinite(v).all():raise ValueError('Nonfinite FP16 cache')
            path=folder/f'layer{layer:02d}.safetensors'
            save_file({'key':k.contiguous(),'value':v.contiguous()},str(path))
            layers.append(dict(layer=layer,path=str(path.resolve()),sha256=digest(path)))
        manifest['splits'][split]=dict(tokens=tokens,context_sha256=sources[split]['context_sha256'],
            token_ids_path=str(token_path.resolve()),token_ids_sha256=digest(token_path),layers=layers)
        write_json(output/'manifest.json',manifest)
        del past,ids,k,v
        if torch.cuda.is_available():torch.cuda.empty_cache()
    manifest['status']='complete';write_json(output/'manifest.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(os.environ.get('QWEN35_MODEL_DIR','/data0/models/qwen35-9b')))
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('INFINITEBENCH_DATA_ROOT','/data0/work/InfiniteBench/data')))
    p.add_argument('--fit-tokens',type=int,default=16384);p.add_argument('--eval-tokens',type=int,default=4096)
    p.add_argument('--prefill-chunk',type=int,default=1024);p.add_argument('--device',default='auto')
    p.add_argument('--model-dtype',choices=['float16','bfloat16'],default='bfloat16')
    a=p.parse_args()
    if min(a.fit_tokens,a.eval_tokens,a.prefill_chunk)<=0:raise ValueError('Lengths must be positive')
    collect(a.output,a.model,a.data_root,a.fit_tokens,a.eval_tokens,a.prefill_chunk,a.device,a.model_dtype)

if __name__=='__main__':main()
