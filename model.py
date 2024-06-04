import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# %%
@dataclass
class GPTConfig:
    block_size = 64
    vocab_size = 300
    n_embed = 100
    n_heads = 5
    n_layers = 2
    dropout = 0
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.head_size = config.n_embed // config.n_heads
        self.register_buffer('tril',torch.tril(
            torch.ones(config.block_size,config.block_size)))
        self.attention = nn.Linear(config.n_embed,3*config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.final_lin_proj = nn.Linear(config.n_embed,config.n_embed)
        self.final_dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        x = self.attention(x)
        K, Q, V = x.split(self.n_embed, dim=-1)
        K = K.view(B, T, -1, self.head_size).transpose(1,2)
        Q = Q.view(B, T, -1, self.head_size).transpose(1,2)
        V = V.view(B, T, -1, self.head_size).transpose(1,2)
        
        weights = Q @ K.transpose(-1,-2) * self.head_size**-0.5
        weights = weights.masked_fill(self.tril==0,float('-inf'))
        weights = torch.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        x = weights @ V 
        x = x.transpose(2,1).reshape(B,T,self.n_embed)
        x = self.final_lin_proj(x)
        x = self.final_dropout(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin = nn.Linear(config.n_embed,4*config.n_embed)
        self.gelu = nn.GELU()
        self.lin_proj = nn.Linear(4*config.n_embed,config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.lin(x)
        x = self.gelu(x)
        x = self.lin_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        
    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    
class WieszczGPT(nn.Module):
    def __init__(self, config=GPTConfig):
        super().__init__()
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tok_embed = nn.Embedding(config.vocab_size,config.n_embed)
        self.pos_embed = nn.Embedding(config.block_size,config.n_embed)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.linear = nn.Linear(config.n_embed,config.vocab_size)
        self.apply(self._initialize_weights)

        for p, w in self.named_parameters():
            if p.endswith('lin_proj.weight'):
                nn.init.normal_(w, 0,0.02 / (config.n_layers**0.5))

    def _initialize_weights(self, module):
        if isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,0,0.02)
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight,0,0.02)
            if not module.bias is None:
                nn.init.zeros_(module.bias)
        if isinstance(module,nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
            

    def forward(self, x, targets=None):
        device = x.device
        x = self.tok_embed(x) + self.pos_embed(torch.arange(self.config.block_size, device=device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.linear(x)
        
        if targets is None:
            loss = None
        else:
            a,b,c = logits.shape
            loss = F.cross_entropy(logits.view(a*b,c),targets.view(a*b))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        return idx