import torch
import torch.nn as nn
from model.TANR.attention import Attention

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
        title_CNN = [
            nn.Conv2d(1, args.n_filters, kernel_size=(args.window_size, args.word_embedding_dim),
                      padding=((args.window_size - 1) // 2, 0)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=args.dropout_ratio, inplace=False)
        ]
        self.title_CNN = nn.Sequential(*title_CNN)
        self.title_attention = Attention(args.query_vector_dim, args.n_filters)

    def forward(self, news):
        '''
        Args:
            news:
                {
                    'title': Tensor(batch_size) * n_words_title
                    'abstract': Tensor(batch_size) * n_words_abstract
                    'body': Tensor(batch_size) * n_words_body
                }
        Returns:
            news_vector: Tensor(batch_size, n_filters)
        '''

        # title
        # torch.stack(news['title'], dim=1) size: (batch_size, n_words_title)
        # batch_size, n_words_title, word_embedding_dim
        title_embedded = self.word_embedding(torch.stack(news['title'], dim=1).to(self.args.device))
        # batch_size, n_filters, n_words_title
        title_feature = self.title_CNN(title_embedded.unsqueeze(dim=1)).squeeze(dim=3)
        title_vector = self.title_attention(title_feature.transpose(1, 2))
        return title_vector