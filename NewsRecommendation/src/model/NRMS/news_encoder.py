import torch
import torch.nn as nn
from model.NRMS.multi_head import MultiHeadSelfAttention
from model.NRMS.additive import AdditiveAttention

class NewsEncoder(torch.nn.Module):
    def __init__(self, args, word_embedding):
        super(NewsEncoder, self).__init__()
        self.args = args
        # title
        if word_embedding is None:
            self.word_embedding = [
                nn.Embedding(args.n_words, args.word_embedding_dim, padding_idx=0),
                nn.Dropout(p=args.dropout_ratio, inplace=False)
            ]
        else:
            self.word_embedding = [
                nn.Embedding.from_pretrained(word_embedding, freeze=False, padding_idx=0),
                nn.Dropout(p=args.dropout_ratio, inplace=False)
            ]
        self.word_embedding = nn.Sequential(*self.word_embedding)
        self.multi_head_self_attention = MultiHeadSelfAttention(args.n_heads, args.word_embedding_dim)
        self.additive_attention = AdditiveAttention(args.query_vector_dim, args.word_embedding_dim)

    def forward(self, news):
        '''
        Args:
            news:
                {
                    'title': Tensor(batch_size) * n_words_title
                }
        Returns:
            news_vector: Tensor(batch_size, word_embedding_dim)
        '''

        # Title
        # torch.stack(news['title'], dim=1) size: (batch_size, n_words_title)
        # batch_size, n_words_title, word_embedding_dim
        title_embedded = self.word_embedding(torch.stack(news['title'], dim=1).to(self.args.device))
        h = self.multi_head_self_attention(title_embedded)
        news_vector = self.additive_attention(h)
        return news_vector
