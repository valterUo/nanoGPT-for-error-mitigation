import torch.nn as nn
from karateclub.graph_embedding import Graph2Vec

class EmbeddingWithAugmentation(nn.Module):
    def __init__(self, config):

        super().__init__()

        # Compute graph embeddings from scratch
        self.vectorizer = Graph2Vec(wl_iterations = 2, 
                       dimensions = config.embedding_dim,
                       workers = 8, 
                       down_sampling = 0.0001, 
                       epochs = 10, 
                       learning_rate = 0.025, 
                       min_count = 1, 
                       seed = 0, 
                       erase_base_features = False)

        #graph_embeddings_tensor = torch.tensor(config.graph_embeddings, dtype=torch.float32)
        #assert config.embedding_dim == graph_embeddings_tensor.shape[1]
        #self.embedding = nn.Embedding.from_pretrained(graph_embeddings_tensor, freeze=True)

        # Additional layers for embedding augmentation
        self.augment_layers = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )

    def forward(self, graphs):
        self.vectorizer.fit(graphs)    
        embeddings = self.vectorizer.get_embedding()
        augmented_embeds = self.augment_layers(embeddings)
        return augmented_embeds
