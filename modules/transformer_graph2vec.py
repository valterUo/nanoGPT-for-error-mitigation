import torch
import torch.nn as nn
import math

from modules.augmented_embedding import EmbeddingWithAugmentation
from modules.decoder import Decoder
from modules.encoder import Encoder
from modules.layernorm import LayerNorm

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerWithGraph2VecEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        #self.positional_encoder = SinusoidalPositionalEncoding(config.n_embd)
        #self.positional_decoder = SinusoidalPositionalEncoding(config.n_embd)

        #self.graph_embedder = EmbeddingWithAugmentation(config)
        
        self.wte_encoder_continuous = nn.Linear(config.graph_input_embedding, config.n_embd, dtype=config.precision)
        #self.wpe_encoder = nn.Embedding(config.block_size, config.n_embd)

        self.wte_decoder_continuous = nn.Linear(config.batch_size, config.n_embd, dtype=config.precision)
        #self.wpe_decoder = nn.Embedding(config.block_size, config.n_embd)
        
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias, config=config)

    def forward(self, src, graph_embeddings, src_mask=None, tgt_mask=None):

        #print(f"src: {src.shape}")
        #print(f"graph_embeddings: {graph_embeddings.shape}")

        src_emb = self.wte_encoder_continuous(graph_embeddings)
        #print(f"src_emb: {src_emb.shape}")
        #src_emb = self.positional_encoder(src_emb)

        #src_emb += self.wpe_encoder(torch.arange(0, graph_embeddings.size(0), device=src.device))
        #print(f"src_emb: {src_emb.shape}")
        
        src_emb = self.drop(src_emb)
        encoder_output = self.encoder(src_emb, mask=src_mask)

        tgt_emb = self.wte_decoder_continuous(src)
        #tgt_emb = self.positional_decoder(tgt_emb)
        #tgt_emb += self.wpe_decoder(torch.arange(0, src.size(1), device=src.device))
        tgt_emb = self.drop(tgt_emb)
        decoder_output = self.decoder(tgt_emb, encoder_output, mask=tgt_mask)

        return self.ln_f(decoder_output)
