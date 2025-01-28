import torch.nn as nn

class EmbeddingWithAugmentation(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.augment_layers = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.n_embd),
        )

    def forward(self, embeddings):
        augmented_embeds = self.augment_layers(embeddings)
        return augmented_embeds
