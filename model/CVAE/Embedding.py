import torch.nn as nn
import numpy as np
import torch
from utils import config


class Embedding(nn.Module):
    def __init__(self, vocab,
                 embedding_size,
                 pad_id=0,
                 dropout=0.1):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.pad_id = pad_id
        self.vocab = vocab
        self.embedding = self.share_embedding(config.preptrained)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # [batch, seq]
        return self.dropout(self.embedding(x))  # [batch, seq, embedding_size]

    def gen_embeddings(self):
        """
            Generate an initial embedding matrix for `word_dict`.
            If an embedding file is not given or a word is not in the embedding file,
            a randomly initialized vector will be used.
        """
        embeddings = np.random.randn(self.vocab.n_words, config.emb_dim) * 0.01
        print('Embeddings: %d x %d' % (self.vocab.n_words, config.emb_dim))
        if config.emb_file is not None:
            print('Loading embedding file: %s' % config.emb_file)
            pre_trained = 0
            for line in open(config.emb_file).readlines():
                sp = line.split()
                if (len(sp) == config.emb_dim + 1):
                    if sp[0] in self.vocab.word2index:
                        pre_trained += 1
                        embeddings[self.vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
                else:
                    print(sp[0])
            print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / self.vocab.n_words))
        return embeddings

    def share_embedding(self, pretrain=True):
        embedding = nn.Embedding(self.vocab.n_words, self.embedding_size, self.pad_id)
        if (pretrain):
            pre_embedding = self.gen_embeddings()
            embedding.weight.data.copy_(torch.FloatTensor(pre_embedding))
            embedding.weight.data.requires_grad = True
        return embedding
