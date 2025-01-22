import torch.nn as nn

from modules.cross_attention import CrossAttention
from modules.layernorm import LayerNorm
from modules.mlp import MLP
from modules.self_attention import CausalSelfAttention

class Block(nn.Module):
    def __init__(self, config, is_decoder=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # Additional components for decoder
        if is_decoder:
            self.cross_attn = CrossAttention(config)
            self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, encoder_output=None, mask=None):
        # Self-attention
        x = x + self.attn(self.ln_1(x), mask=mask)
        
        # Cross-attention (only in decoder)
        if encoder_output is not None:
            x = x + self.cross_attn(self.ln_2(x), encoder_output, mask=mask)
            x = self.ln_3(x)  # Apply layer norm after cross-attention

        # Feed-forward network
        x = x + self.mlp(self.ln_2(x))
        return x