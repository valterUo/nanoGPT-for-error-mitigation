import torch.nn as nn
from modules.block import Block
from modules.layernorm import LayerNorm

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Block(config, is_decoder=True) for _ in range(config.n_layer)])
        self.ln = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, encoder_output, mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output=encoder_output, mask=mask)
        x = self.ln(x)
        return x