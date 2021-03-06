import torch
import torch.nn as nn
from model.NAML.attention import Attention
from model.NAML.fm import FM_Layer

class NewsEncoder(torch.nn.Module):
    def __init__(self, args, word_embedding):
        super(NewsEncoder, self).__init__()
        self.args = args

        # categories(topic)
        self.category_embedding = nn.Embedding(args.n_categories, args.category_embedding_dim, padding_idx=0)
        category_dense = [
            nn.Linear(args.category_embedding_dim, args.n_filters),
            nn.ReLU(inplace=False)
        ]
        self.category_dense = nn.Sequential(*category_dense)
        # subcategories
        subcategory_dense = [
            nn.Linear(args.category_embedding_dim, args.n_filters),
            nn.ReLU(inplace=False)
        ]
        self.subcategory_dense = nn.Sequential(*subcategory_dense)
        # hot_click
        self.hot_click_embedding = nn.Embedding(args.n_hot_clicks, args.category_embedding_dim, padding_idx=0)
        hot_click_dense = [
            nn.Linear(args.category_embedding_dim, args.n_filters),
            nn.ReLU(inplace=False)
        ]
        self.hot_click_dense = nn.Sequential(*hot_click_dense)
        # hot_display
        self.hot_display_embedding = nn.Embedding(args.n_hot_displays, args.category_embedding_dim, padding_idx=0)
        hot_display_dense = [
            nn.Linear(args.category_embedding_dim, args.n_filters),
            nn.ReLU(inplace=False)
        ]
        self.hot_display_dense = nn.Sequential(*hot_display_dense)
        # burst_click
        self.burst_click_embedding = nn.Embedding(args.n_burst_clicks, args.category_embedding_dim, padding_idx=0)
        burst_click_dense = [
            nn.Linear(args.category_embedding_dim, args.n_filters),
            nn.ReLU(inplace=False)
        ]
        self.burst_click_dense = nn.Sequential(*burst_click_dense)
        # burst_display
        self.burst_display_embedding = nn.Embedding(args.n_burst_displays, args.category_embedding_dim, padding_idx=0)
        burst_display_dense = [
            nn.Linear(args.category_embedding_dim, args.n_filters),
            nn.ReLU(inplace=False)
        ]
        self.burst_display_dense = nn.Sequential(*burst_display_dense)
        # title, abstract, body
        if word_embedding is None:
            self.word_embedding = [
                nn.Embedding(args.n_words, args.word_embedding_dim, padding_idx=0),
                nn.Dropout(p=args.dropout_ratio, inplace=False)
            ]
            print('nuse nn.Embedding...')
        else:
            self.word_embedding = [
                nn.Embedding.from_pretrained(word_embedding, freeze=False, padding_idx=0),
                nn.Dropout(p=args.dropout_ratio, inplace=False)
            ]
            print('use nn.Embedding.from_pretrained...')
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

        # print('fm_input.shape:', fm_input.shape) # fm_input.shape: torch.Size([64, 400, 6])
        self.fm = FM_Layer(n=6, k=1)


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
        # cnt_test = 0
        # if cnt_test == 0:
        #     print('self.args.device:', self.args.device)
        #     cnt_test += 1
        # batch_size, n_filters
        category_vector = self.category_dense(category_embedded)
        # print('category_vector.shape:', category_vector.shape) # torch.Size([64, 400])
        # subcategory
        subcategory_embedded = self.category_embedding(news['subcategory'].to(self.args.device))
        subcategory_vector = self.subcategory_dense(subcategory_embedded)
        # # hot
        # label_hot = [news['hot'][0], news['hot'][1]]
        # one_hot_hot = torch.zeros(1, self.args.n_hots).scatter_(1, label_hot, 1)
        # hot_embedded = self.hot_embedding(one_hot_hot.to(self.args.device))
        # hot_vector = self.hot_dense(hot_embedded)
        hot_click_embedded = self.hot_click_embedding(news['hot_click'].to(self.args.device))
        hot_click_vector = self.hot_click_dense(hot_click_embedded)
        hot_display_embedded = self.hot_display_embedding(news['hot_display'].to(self.args.device))
        hot_display_vector = self.hot_display_dense(hot_display_embedded)
        burst_click_embedded = self.burst_click_embedding(news['burst_click'].to(self.args.device))
        burst_click_vector = self.burst_click_dense(burst_click_embedded)
        burst_display_embedded = self.burst_display_embedding(news['burst_display'].to(self.args.device))
        burst_display_vector = self.burst_display_dense(burst_display_embedded)
        # print('burst_display_vector.shape:', burst_display_vector.shape) # torch.Size([64, 400])

        fm_input = torch.stack([category_vector, subcategory_vector, hot_click_vector, hot_display_vector,
                                burst_click_vector, burst_display_vector], dim=2).to(self.args.device)

        yq_vector = self.fm(fm_input)
        yq_vector = yq_vector.squeeze()
        # print('yq_vector.shape:', yq_vector.shape)


        # title
        # torch.stack(news['title'], dim=1) size: (batch_size, n_words_title)
        # batch_size, n_words_title, word_embedding_dim
        title_embedded = self.word_embedding(torch.stack(news['title'], dim=1).to(self.args.device))
        # batch_size, n_filters, n_words_title
        title_feature = self.title_CNN(title_embedded.unsqueeze(dim=1)).squeeze(dim=3)
        title_vector = self.title_attention(title_feature.transpose(1, 2))
        # print('title_vector.shape:', title_vector.shape)
        # abstract
        abstract_embedded = self.word_embedding(torch.stack(news['abstract'], dim=1).to(self.args.device))
        abstract_feature = self.abstract_CNN(abstract_embedded.unsqueeze(dim=1)).squeeze(dim=3)
        abstract_vector = self.abstract_attention(abstract_feature.transpose(1, 2))
        # print('abstract_vector.shape:', abstract_vector.shape)
        # body
        body_embedded = self.word_embedding(torch.stack(news['body'], dim=1).to(self.args.device))
        body_feature = self.body_CNN(body_embedded.unsqueeze(dim=1)).squeeze(dim=3)
        body_vector = self.body_attention(body_feature.transpose(1, 2))
        # print('body_vector.shape:', body_vector.shape)
        # total
        news_vector = self.total_attention(
            torch.stack([yq_vector, title_vector, abstract_vector, body_vector], dim=1))
        return news_vector