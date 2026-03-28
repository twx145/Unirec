from __future__ import annotations
import hashlib, torch

TYPE_TO_ID={'unknown':0,'int_value':1,'float_value':2,'int_array':3,'float_array':4,'int_array_and_float_array':5}
SEQ_NAME_TO_ID={'action_seq':1,'content_seq':2,'item_seq':3}
SPECIAL={'user_id':1,'item_id':2,'timestamp_bucket':3}

def _h(s, mod): return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:12],16)%(mod-1)+1

class TAACCollator:
    def __init__(self,cfg):
        dc,mc=cfg['data'],cfg['model']
        self.hash_size=int(mc['hash_size']); self.max_static=int(dc['max_static_tokens']); self.max_float=int(dc['max_float_dim']); self.max_seq=int(dc['max_seq_len']); self.max_event=int(dc['max_event_features']); self.sequence_names=list(dc.get('sequence_names',['action_seq','content_seq','item_seq']))
    def _float_vals(self, feat):
        arr=feat.get('float_array')
        vals=[float(x) for x in (arr[:self.max_float] if arr is not None else ([feat['float_value']] if feat.get('float_value') is not None else []))]
        return vals + [0.0]*(self.max_float-len(vals))
    def _stat_tok(self, ns, feat):
        fid=int(feat.get('feature_id',0)); typ=feat.get('feature_value_type') or 'unknown'; tid=TYPE_TO_ID.get(typ,0)
        if feat.get('int_value') is not None: key=str(int(feat['int_value']))
        elif feat.get('int_array') is not None: key=','.join(map(str, feat['int_array'][:8]))
        elif feat.get('float_array') is not None or feat.get('float_value') is not None: key='float'
        else: key='null'
        tok=_h(f'{ns}|{fid}|{typ}|{key}', self.hash_size)
        return tok,fid,tid,self._float_vals(feat)
    def _special(self, name, val):
        return _h(f'{name}|{val}',self.hash_size), SPECIAL[name], 0, [0.0]*self.max_float
    def _build_static(self,s):
        toks=[self._special('user_id',s.get('user_id','')), self._special('item_id',s.get('item_id',0)), self._special('timestamp_bucket',int(s.get('timestamp',0))//3600)]
        toks += [self._stat_tok('user',f) for f in s.get('user_feature',[]) or []]
        toks += [self._stat_tok('item',f) for f in s.get('item_feature',[]) or []]
        toks=toks[:self.max_static]
        ids=torch.zeros(self.max_static,dtype=torch.long); fids=torch.zeros_like(ids); tids=torch.zeros_like(ids); fvals=torch.zeros(self.max_static,self.max_float); mask=torch.zeros(self.max_static,dtype=torch.bool)
        for i,(a,b,c,d) in enumerate(toks): ids[i]=a; fids[i]=b; tids[i]=c; fvals[i]=torch.tensor(d); mask[i]=True
        return ids,fids,tids,fvals,mask
    def _build_seq(self, name, payload):
        feats=payload or []; L=0
        for f in feats:
            arr=f.get('int_array') if isinstance(f,dict) else None
            if arr is not None: L=max(L,len(arr))
        L=min(L,self.max_seq)
        ids=torch.zeros(self.max_seq,self.max_event,dtype=torch.long); fids=torch.zeros_like(ids); tids=torch.zeros_like(ids); pos=torch.arange(self.max_seq,dtype=torch.long); mask=torch.zeros(self.max_seq,dtype=torch.bool); sid=torch.full((self.max_seq,), SEQ_NAME_TO_ID.get(name,0), dtype=torch.long)
        start=max(0,L-self.max_seq)
        for o,p in enumerate(range(start,L)):
            k=0
            for f in feats:
                arr=f.get('int_array') if isinstance(f,dict) else None
                if arr is None or p>=len(arr) or k>=self.max_event: continue
                fid=int(f.get('feature_id',0)); typ=f.get('feature_value_type') or 'int_array'; tid=TYPE_TO_ID.get(typ,3); val=int(arr[p]); ids[o,k]=_h(f'seq:{name}|{fid}|{typ}|{val}', self.hash_size); fids[o,k]=fid; tids[o,k]=tid; k+=1
            if k>0: mask[o]=True
        return ids,fids,tids,pos,mask,sid
    def __call__(self,batch):
        out={k:[] for k in ['static_token_ids','static_feature_ids','static_type_ids','static_float_values','static_mask','seq_token_ids','seq_feature_ids','seq_type_ids','seq_pos_ids','seq_mask','seq_name_ids']}
        targets=torch.tensor([float(x.get('target',0.0)) for x in batch],dtype=torch.float32)
        item_ids=torch.tensor([int(x.get('item_id',0) or 0) for x in batch],dtype=torch.long)
        for s in batch:
            a,b,c,d,e=self._build_static(s)
            out['static_token_ids'].append(a); out['static_feature_ids'].append(b); out['static_type_ids'].append(c); out['static_float_values'].append(d); out['static_mask'].append(e)
            seq=s.get('seq_feature',{}) or {}
            allv=[self._build_seq(n, seq.get(n,[]) if isinstance(seq,dict) else []) for n in self.sequence_names]
            out['seq_token_ids'].append(torch.stack([x[0] for x in allv])); out['seq_feature_ids'].append(torch.stack([x[1] for x in allv])); out['seq_type_ids'].append(torch.stack([x[2] for x in allv])); out['seq_pos_ids'].append(torch.stack([x[3] for x in allv])); out['seq_mask'].append(torch.stack([x[4] for x in allv])); out['seq_name_ids'].append(torch.stack([x[5] for x in allv]))
        out={k:torch.stack(v) for k,v in out.items()}
        out['targets']=targets; out['item_ids']=item_ids
        return out
