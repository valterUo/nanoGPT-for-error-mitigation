import torch
import torch.nn as nn

from modules.decoder import Decoder
from modules.encoder import Encoder
from modules.layernorm import LayerNorm

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.wte_encoder = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe_encoder = nn.Embedding(config.block_size, config.n_embd)
        self.wte_decoder = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe_decoder = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed and encode the source sequence
        
        # This would be the standard embedding if the input for the encoder is just tokens
        src_emb = self.wte_encoder(src) + self.wpe_encoder(torch.arange(0, src.size(1), device=src.device))
        src_emb = self.drop(src_emb)
        
        encoder_output = self.encoder(src_emb, mask=src_mask)

        # Embed and decode the target sequence
        tgt_emb = self.wte_decoder(tgt) + self.wpe_decoder(torch.arange(0, tgt.size(1), device=tgt.device))
        tgt_emb = self.drop(tgt_emb)
        decoder_output = self.decoder(tgt_emb, encoder_output, mask=tgt_mask)

        return self.ln_f(decoder_output)