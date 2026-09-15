"""KVCodec-style token frames, symmetric INT8, lossless FFmpeg HEVC.
No rotations or centering. Wire scales are BF16, rounded upward to cover range.
"""
import json,struct,subprocess
import numpy as np

TOKENS_PER_FRAME=8

def quantize(xs,epsilon):
    x=np.concatenate(xs,axis=0).astype(np.float32)
    error=np.sqrt(12.)*epsilon*np.sqrt(np.mean(x.astype(np.float64)**2,axis=(1,2)))
    limit=np.max(np.abs(x),axis=(1,2)).astype(np.float64)/127
    target=np.maximum(error,limit);target=np.where(target>0,target,1.)
    bits=target.astype(np.float32).view(np.uint32)
    bits=((bits.astype(np.uint64)+65535)&0xffff0000).astype(np.uint32)
    bits+=np.where(bits.view(np.float32).astype(np.float64)<target,65536,0).astype(np.uint32)
    scales=bits.view(np.float32)
    q=np.rint(x/scales[:,None,None])
    if np.any(np.abs(q)>127):raise ValueError('INT8 range overflow')
    return (q.astype(np.int16)+128).astype(np.uint8),scales,int(np.sum(limit>=error))

def layout(q,model,tokens_per_frame=TOKENS_PER_FRAME):
    if tokens_per_frame<1:raise ValueError('tokens_per_frame must be positive')
    if model=='qwen':
        if q.shape[0]!=64 or q.shape[2]!=256:raise ValueError('Qwen geometry')
        q=np.concatenate((q[:32],q[32:]),axis=2)
    views,tokens,width=q.shape
    padded_views={'qwen':64,'glm':16,'deepseek':32}.get(model)
    if padded_views is None or views>padded_views:raise ValueError('Unknown model/view geometry')
    n=(tokens+tokens_per_frame-1)//tokens_per_frame
    packed=np.full((n,tokens_per_frame,padded_views,width),128,np.uint8)
    for i in range(tokens):packed[i//tokens_per_frame,i%tokens_per_frame,:views]=q[:,i]
    return packed.reshape(n,tokens_per_frame*padded_views,width)

def inverse(y,model,views,tokens,tokens_per_frame=TOKENS_PER_FRAME):
    view_rows=32 if model=='qwen' else views
    width=y.shape[2]
    padded_views={'qwen':64,'glm':16,'deepseek':32}.get(model)
    if padded_views is None:raise ValueError('Unknown model geometry')
    q=y[:,:padded_views*tokens_per_frame].reshape(y.shape[0],tokens_per_frame,padded_views,width)
    q=q[:,:,:view_rows].transpose(2,0,1,3).reshape(view_rows,-1,width)[:,:tokens]
    if model=='qwen':q=np.concatenate((q[:,:,:256],q[:,:,256:]),axis=0)
    return q.copy()

def parameters(n,h):
    # Same holdout cell as kvpipe.e1_e2_rd.holdout_x265_params; one worker per process.
    return ':'.join(['lossless=1','scenecut=0','bframes=3','b-adapt=2','open-gop=0','log-level=error',
        f'keyint={n}',f'min-keyint={n}','pools=1','frame-threads=1','wpp=1','pmode=0',
        'rskip=0','early-skip=0','fast-intra=0','signhide=0','sao=0','strong-intra-smoothing=0',
        'weightp=0','weightb=0','aq-mode=0','cutree=0','rd=3','subme=3','psy-rd=0','psy-rdoq=0',
        'ctu='+str(16 if h<32 else 32 if h<64 else 64),'range=full','annexb=1','repeat-headers=1','hash=1','rc-lookahead=4'])

def run(cmd,data):
    p=subprocess.run(cmd,input=data,capture_output=True,timeout=300)
    if p.returncode or not p.stdout:raise RuntimeError('FFmpeg failed: '+p.stderr.decode(errors='replace')[-4000:])
    return p.stdout

def encode(xs,model,epsilon,ffmpeg):
    q,scales,floors=quantize(xs,epsilon);tokens=q.shape[1];y=layout(q,model);n,h,w=y.shape
    raw=np.concatenate((y.reshape(n,-1),np.full((n,h*w//2),128,np.uint8)),axis=1).tobytes()
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-threads','1','-filter_threads','1','-f','rawvideo','-pix_fmt','yuv420p','-video_size',f'{w}x{h}','-framerate','1','-i','pipe:0','-frames:v',str(n),'-an','-c:v','libx265','-preset','veryfast','-x265-params',parameters(n,h),'-threads','1','-pix_fmt','yuv420p','-f','hevc','pipe:1']
    payload=run(cmd,raw)
    meta=dict(format='kvcodec-token-hevc-v2',model=model,frames=n,height=h,width=w,views=len(scales),groups=[len(x) for x in xs],tokens=tokens,tokens_per_frame=TOKENS_PER_FRAME,range_floor_views=floors)
    head=json.dumps(meta,separators=(',',':')).encode();sb=(scales.view(np.uint32)>>16).astype('<u2').tobytes()
    return struct.pack('<I',len(head))+head+sb+payload

def parse(packet):
    n,=struct.unpack_from('<I',packet);head=json.loads(packet[4:4+n]);pos=4+n
    if head['format'] not in ('kvcodec-token-hevc-v1','kvcodec-token-hevc-v2'):raise ValueError('Wrong packet')
    scales=(np.frombuffer(packet,dtype='<u2',count=head['views'],offset=pos).astype(np.uint32)<<16).view(np.float32)
    pos+=head['views']*2
    return head,scales,packet[pos:],pos

def decode(packet,ffmpeg):
    h,s,p,_=parse(packet)
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-threads','1','-filter_threads','1','-f','hevc','-i','pipe:0','-frames:v',str(h['frames']),'-vf','extractplanes=y','-pix_fmt','gray','-threads','1','-f','rawvideo','pipe:1']
    raw=run(cmd,p);y=np.frombuffer(raw,dtype=np.uint8).reshape(h['frames'],h['height'],h['width'])
    tokens=h.get('tokens',h['frames']*h.get('tokens_per_frame',1))
    return inverse(y,h['model'],h['views'],tokens,h.get('tokens_per_frame',1)),s
