import torch
import torch.nn as nn
from model.NAML.attention import Attention

class NewsEncoder(torch.nn.Module):
    def __init__(self, args, word_embedding):
        super(NewsEncoder, self).__init__()
        self.args = args
        # categories
        self.category_embedding = nn.Embedding(args.n_categories, args.category_embedding_dim, padding_idx=0)
        category_dense = [
            nn.Linear(args.category_embedding_dim, args.n_filters),
            nn.ReLU(inplace=False)
        ]
        self.category_dense = nn.Sequential(*category_dense)
        subcategory_dense = [
            nn.Linear(args.category_embedding_dim, args.n_filters),
            nn.ReLU(inplace=False)
        ]
        self.subcategory_dense = nn.Sequential(*subcategory_dense)
        # title, abstract and body
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
        abstract_CNN = [
            nn.Conv2d(1, args.n_filters, kernel_size=(args.window_size, args.word_embedding_dim),
                      padding=((args.window_size - 1) // 2, 0)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=args.dropout_ratio, inplace=False)
        ]
        self.abstract_CNN = nn.Sequential(*abstract_CNN)
        body_CNN = [
            nn.Conv2d(1, args.n_filters, kernel_size=(args.window_size, args.word_embedding_dim),
                      padding=((args.window_size - 1) // 2, 0)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=args.dropout_ratio, inplace=False)
        ]
        self.body_CNN = nn.Sequential(*body_CNN)
        self.title_attention = Attention(args.query_vector_dim, args.n_filters)
        self.abstract_attention = Attention(args.query_vector_dim, args.n_filters)
        self.body_attention = Attention(args.query_vector_dim, args.n_filters)
        self.total_attention = Attention(args.query_vector_dim, args.n_filters)

    def forward(self, news):
        '''
        Args:
            news:
                {
                    'category': Tensor(batch_size),
                    'subcategory': Tensor(batch_size),
                    'title': Tensor(batch_size) * n_words_title
                    'abstract': Tensor(batch_size) * n_words_abstract
                    'body': Tensor(batch_size) * n_words_body
                }
        Returns:
            news_vector: Tensor(batch_size, n_filters)
        '''

        # Categories
        # batch_size, category_embedding_dim
        category_embedded = self.category_embedding(news['category'].to(self.args.device))
        # batch_size, n_filters
        category_vector = self.category_dense(category_embedded)
        subcategory_embedded = self.category_embedding(news['subcategory'].to(self.args.device))
        subcategory_vector = self.subcategory_dense(subcategory_embedded)

        # title
        # torch.stack(news['title'], dim=1) size: (batch_size, n_words_title)
        # batch_size, n_words_title, word_embedding_dim
        title_embedded = self.word_embedding(torch.stack(news['title'], dim=1).to(self.args.device))
        # batch_size, n_filters, n_words_title
        title_feature = self.title_CNN(title_embedded.unsqueeze(dim=1)).squeeze(dim=3)
        title_vector = self.title_attention(title_feature.transpose(1, 2))
        # abstract
        abstract_embedded = self.word_embedding(torch.stack(news['abstract'], dim=1).to(self.args.device))
        abstract_feature = self.abstract_CNN(abstract_embedded.unsqueeze(dim=1)).squeeze(dim=3)
        abstract_vector = self.abstract_attention(abstract_feature.transpose(1, 2))
        # body
        body_embedded = self.word_embedding(torch.stack(news['body'], dim=1).to(self.args.device))
        body_feature = self.body_CNN(body_embedded.unsqueeze(dim=1)).squeeze(dim=3)
        body_vector = self.body_attention(body_feature.transpose(1, 2))
        # total
        news_vector = self.total_attention(
            torch.stack([category_vector, subcategory_vector, title_vector, abstract_vector, body_vector], dim=1))
        return news_vector
