"""
Embeddings for the VAE model
"""

import torch.nn as nn


class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their embeddings.
    """
    def __init__(self, embed_size, vocab, padding_idx):
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(len(vocab.vocab_entry), embed_size, padding_idx=padding_idx)



