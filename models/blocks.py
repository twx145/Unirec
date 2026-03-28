import torch, torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x):
        return x*torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)*self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__(); self.w1=nn.Linear(dim,hidden_dim); self.w2=nn.Linear(dim,hidden_dim); self.out=nn.Linear(hidden_dim,dim); self.dropout=nn.Dropout(dropout)
    def forward(self,x): return self.out(self.dropout(torch.nn.functional.silu(self.w1(x))*self.w2(x)))

class SimpleMHA(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.0):
        super().__init__(); assert d_model%n_heads==0; self.d_model=d_model; self.n_heads=n_heads; self.h=d_model//n_heads; self.q=nn.Linear(d_model,d_model); self.k=nn.Linear(d_model,d_model); self.v=nn.Linear(d_model,d_model); self.o=nn.Linear(d_model,d_model); self.dropout=nn.Dropout(dropout)
    def forward(self,q,k,v,kv_mask=None):
        B,Q,_=q.shape; K=k.shape[1]
        q=self.q(q).view(B,Q,self.n_heads,self.h).transpose(1,2)
        k=self.k(k).view(B,K,self.n_heads,self.h).transpose(1,2)
        v=self.v(v).view(B,K,self.n_heads,self.h).transpose(1,2)
        score=(q@k.transpose(-2,-1))/(self.h**0.5)
        if kv_mask is not None: score=score.masked_fill(~kv_mask[:,None,None,:].bool(), -1e4)
        attn=self.dropout(score.softmax(dim=-1))
        out=(attn@v).transpose(1,2).contiguous().view(B,Q,self.d_model)
        return self.o(out)

class CrossAttentionBlock(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.0):
        super().__init__(); self.nq=RMSNorm(d_model); self.nkv=RMSNorm(d_model); self.attn=SimpleMHA(d_model,n_heads,dropout); self.drop=nn.Dropout(dropout)
    def forward(self,q,kv,kv_mask=None): return q + self.drop(self.attn(self.nq(q), self.nkv(kv), self.nkv(kv), kv_mask))

class SelfAttentionBlock(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.0):
        super().__init__(); self.norm=RMSNorm(d_model); self.attn=SimpleMHA(d_model,n_heads,dropout); self.drop=nn.Dropout(dropout)
    def forward(self,x,mask=None): y=self.norm(x); return x + self.drop(self.attn(y,y,y,mask))

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model,hidden_mult=4,dropout=0.0):
        super().__init__(); self.norm=RMSNorm(d_model); self.ff=SwiGLU(d_model,d_model*hidden_mult,dropout); self.drop=nn.Dropout(dropout)
    def forward(self,x): return x + self.drop(self.ff(self.norm(x)))

class TokenMixer(nn.Module):
    def __init__(self,d_model,num_tokens,expansion=2,dropout=0.0):
        super().__init__(); hidden=max(num_tokens,num_tokens*expansion); self.norm=RMSNorm(d_model); self.fc1=nn.Linear(num_tokens,hidden); self.fc2=nn.Linear(hidden,num_tokens); self.drop=nn.Dropout(dropout)
    def forward(self,x): y=self.norm(x).transpose(1,2); y=self.fc2(torch.nn.functional.gelu(self.fc1(y))).transpose(1,2); return x + self.drop(y)

class MemoryCompressor(nn.Module):
    def __init__(self,d_model,memory_tokens,n_heads,dropout=0.0):
        super().__init__(); self.memory=nn.Parameter(torch.randn(1,memory_tokens,d_model)*0.02); self.cross=CrossAttentionBlock(d_model,n_heads,dropout); self.ff=FeedForwardBlock(d_model,dropout=dropout)
    def forward(self,x,mask):
        mem=self.memory.expand(x.size(0),-1,-1)
        return self.ff(self.cross(mem,x,mask))
