import torch, torch.nn as nn, torch.nn.functional as F
from .tokenizer import UnifiedTokenizer
from .blocks import CrossAttentionBlock, SelfAttentionBlock, FeedForwardBlock, TokenMixer, MemoryCompressor

class UniScaleLayer(nn.Module):
    def __init__(self,d_model,n_heads,num_mix_tokens,dropout=0.0):
        super().__init__(); self.cross=CrossAttentionBlock(d_model,n_heads,dropout); self.mix=TokenMixer(d_model,num_mix_tokens,2,dropout); self.ff=FeedForwardBlock(d_model,4,dropout)
    def forward(self,queries,kv,kv_mask,static_tokens):
        q=self.cross(queries,kv,kv_mask)
        x=torch.cat([q, static_tokens], dim=1)
        x=self.ff(self.mix(x))
        return x[:, :queries.size(1)]

class FMHead(nn.Module):
    def forward(self,x):
        s=x.sum(dim=1); return 0.5*((s.pow(2)-x.pow(2).sum(dim=1)).sum(dim=-1,keepdim=True))

class UniScaleFormer(nn.Module):
    def __init__(self,cfg):
        super().__init__(); dc,mc,tc=cfg['data'],cfg['model'],cfg['train']
        self.use_aux=bool(tc.get('use_aux_contrastive',True)); self.aux_weight=float(tc.get('aux_weight',0.05)); self.num_sequences=len(dc.get('sequence_names',['action_seq','content_seq','item_seq'])); self.d_model=int(mc['d_model']); self.num_queries=int(mc['num_queries'])
        self.tokenizer=UnifiedTokenizer(int(mc['hash_size']), int(mc.get('feature_vocab_size',16384)), int(mc.get('type_vocab_size',16)), int(mc.get('seq_vocab_size',16)), int(dc['max_seq_len'])+8, self.d_model, int(dc['max_float_dim']), float(mc['dropout']))
        self.static_encoder=nn.ModuleList([SelfAttentionBlock(self.d_model,int(mc['n_heads']),float(mc['dropout'])) for _ in range(int(mc.get('static_layers',1)))])
        self.seq_encoder=nn.ModuleList([SelfAttentionBlock(self.d_model,int(mc['n_heads']),float(mc['dropout'])) for _ in range(int(mc.get('seq_layers',1)))])
        self.compressors=nn.ModuleList([MemoryCompressor(self.d_model,int(mc['memory_tokens']),int(mc['n_heads']),float(mc['dropout'])) for _ in range(self.num_sequences)])
        self.query_seed=nn.Parameter(torch.randn(1,self.num_queries,self.d_model)*0.02)
        self.query_proj=nn.Sequential(nn.Linear(self.d_model*2,self.d_model), nn.GELU(), nn.Linear(self.d_model,self.d_model))
        self.layers=nn.ModuleList([UniScaleLayer(self.d_model,int(mc['n_heads']), self.num_queries+int(dc['max_static_tokens']), float(mc['dropout'])) for _ in range(int(mc['n_layers']))])
        self.final_attn=SelfAttentionBlock(self.d_model,int(mc['n_heads']),float(mc['dropout']))
        self.fm=FMHead()
        self.head=nn.Sequential(nn.LayerNorm(self.d_model*3+1), nn.Linear(self.d_model*3+1,int(mc['head_hidden'])), nn.GELU(), nn.Dropout(float(mc['dropout'])), nn.Linear(int(mc['head_hidden']),1))
    def _mean(self,x,mask):
        m=mask.float().unsqueeze(-1); return (x*m).sum(dim=1)/m.sum(dim=1).clamp_min(1.0)
    def forward(self,batch):
        st=self.tokenizer.encode_static(batch['static_token_ids'],batch['static_feature_ids'],batch['static_type_ids'],batch['static_float_values'])
        sm=batch['static_mask']
        for blk in self.static_encoder: st=blk(st,sm)
        st_pool=self._mean(st,sm)
        seq=self.tokenizer.encode_sequence_events(batch['seq_token_ids'],batch['seq_feature_ids'],batch['seq_type_ids'],batch['seq_pos_ids'],batch['seq_name_ids'])
        mems=[]; seq_pools=[]
        for s in range(seq.size(1)):
            x=seq[:,s]; m=batch['seq_mask'][:,s]
            for blk in self.seq_encoder: x=blk(x,m)
            mems.append(self.compressors[s](x,m)); seq_pools.append(self._mean(x,m))
        mem=torch.cat(mems,dim=1); mem_mask=torch.ones(mem.shape[:2],dtype=torch.bool,device=mem.device)
        seq_pool=torch.stack(seq_pools,dim=1).mean(dim=1)
        queries=self.query_seed.expand(st.size(0),-1,-1) + self.query_proj(torch.cat([st_pool,seq_pool],dim=-1)).unsqueeze(1)
        for layer in self.layers: queries=layer(queries,mem,mem_mask,st)
        fused=torch.cat([queries,st],dim=1)
        fused_mask=torch.cat([torch.ones(st.size(0),queries.size(1),dtype=torch.bool,device=st.device), sm], dim=1)
        fused=self.final_attn(fused,fused_mask)
        query_pool=queries.mean(dim=1); fused_pool=self._mean(fused,fused_mask); fm=self.fm(torch.cat([queries,st],dim=1))
        logit=self.head(torch.cat([query_pool, st_pool, fused_pool, fm], dim=-1)).squeeze(-1)
        out={'logits':logit, 'probs':torch.sigmoid(logit)}
        if self.use_aux:
            item_embed=self.tokenizer.token_emb(batch['static_token_ids'][:,1].clamp_min(0)); out['aux_similarity']=F.cosine_similarity(query_pool,item_embed,dim=-1)
        return out
    def compute_loss(self,batch,out):
        y=batch['targets']; bce=F.binary_cross_entropy_with_logits(out['logits'], y); loss=bce; aux=torch.tensor(0.0,device=y.device)
        if self.use_aux and 'aux_similarity' in out:
            aux=F.mse_loss(out['aux_similarity'], y*2-1); loss=loss+self.aux_weight*aux
        return {'loss':loss,'bce_loss':bce.detach(),'aux_loss':aux.detach()}
