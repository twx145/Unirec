import torch, torch.nn as nn

class UnifiedTokenizer(nn.Module):
    def __init__(self, hash_size, feature_vocab_size, type_vocab_size, seq_vocab_size, max_position, d_model, max_float_dim, dropout=0.0):
        super().__init__()
        self.token_emb=nn.Embedding(hash_size+1,d_model,padding_idx=0)
        self.feature_emb=nn.Embedding(feature_vocab_size,d_model,padding_idx=0)
        self.type_emb=nn.Embedding(type_vocab_size,d_model,padding_idx=0)
        self.seq_emb=nn.Embedding(seq_vocab_size,d_model,padding_idx=0)
        self.pos_emb=nn.Embedding(max_position,d_model)
        self.float_proj=nn.Sequential(nn.LayerNorm(max_float_dim), nn.Linear(max_float_dim,d_model), nn.GELU(), nn.Linear(d_model,d_model))
        self.dropout=nn.Dropout(dropout)
    def encode_static(self, token_ids, feature_ids, type_ids, float_values):
        x=self.token_emb(token_ids)+self.feature_emb(feature_ids.clamp_min(0))+self.type_emb(type_ids.clamp_min(0))+self.float_proj(float_values)
        return self.dropout(x)
    def encode_sequence_events(self, token_ids, feature_ids, type_ids, pos_ids, seq_name_ids):
        token=self.token_emb(token_ids)+self.feature_emb(feature_ids.clamp_min(0))+self.type_emb(type_ids.clamp_min(0))
        mask=token_ids.ne(0).float().unsqueeze(-1)
        event=(token*mask).sum(dim=3)/mask.sum(dim=3).clamp_min(1.0)
        event=event+self.pos_emb(pos_ids)+self.seq_emb(seq_name_ids)
        return self.dropout(event)
