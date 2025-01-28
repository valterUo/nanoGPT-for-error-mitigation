import torch
import torch.nn as nn

from modules.cross_attention import CrossAttention
from modules.layernorm import LayerNorm
from modules.mlp import MLP
from modules.self_attention import CausalSelfAttention

class Block(nn.Module):
    def __init__(self, config, is_decoder=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias, config=config)
        self.attn = MLP(config) #CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias, config=config)
        self.mlp = MLP(config)
        self.cross_attn_alpha = MLP(config)

        # Additional components for decoder
        if is_decoder:
            self.cross_attn = MLP(config) #CrossAttention(config)
            self.ln_3 = LayerNorm(config.n_embd, bias=config.bias, config=config)
            self.projection_layer = nn.Linear(2 * config.n_embd, config.n_embd, dtype=config.precision)

    def forward(self, x, encoder_output=None, mask=None):
        # Self-attention
        x = x + self.attn(self.ln_1(x)) #, mask=mask)
        
        # Cross-attention (only in decoder)
        if encoder_output is not None:
            #x = self.ln_2(x)
            #x_expanded = x.unsqueeze(0).expand(encoder_output.size(0), -1)
            #print(f"x: {x.shape}")
            #print(f"encoder_output: {encoder_output.shape}")
            
            #concatenated_input = torch.cat((x, encoder_output), dim=1)
            #projected_input = self.projection_layer(concatenated_input)
            
            alpha = torch.sigmoid(self.cross_attn_alpha(self.ln_2(x)))
            interaction = alpha * encoder_output + (1 - alpha) * self.ln_2(x)
            x = x + self.cross_attn(interaction)

            #x = x + self.cross_attn(projected_input) #, mask=mask)
            #x = x + self.ln_2(projected_input)
            x = self.ln_3(x)  # Apply layer norm after cross-attention

        # Feed-forward network
        x = x + self.mlp(self.ln_2(x))
        return x