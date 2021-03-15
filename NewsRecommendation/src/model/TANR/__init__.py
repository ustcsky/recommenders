import torch
import torch.nn as nn
from model.TANR.news_encoder import NewsEncoder
from model.TANR.user_encoder import UserEncoder
from model.TANR.click_predictor import ClickPredictor

def make_model(args, word_embedding):
    return TANR(args, word_embedding)

class TANR(torch.nn.Module):

    def __init__(self, args, word_embedding):
        super(TANR, self).__init__()
        self.args = args
        self.news_encoder = NewsEncoder(args, word_embedding)
        self.user_encoder = UserEncoder(args)
        self.click_predictor = ClickPredictor()
        self.topic_predictor = nn.Linear(args.n_filters, args.n_categories)

    def forward(self, batch):
        # K+1, batch_size, n_filters
        candidate_vector = torch.stack([self.news_encoder(x) for x in batch['candidate']])
        # batch_size, n_browsed_news, n_filters
        browsed_vector = torch.stack([self.news_encoder(x) for x in batch['browsed']], dim=1)
        # batch_size, n_filters
        user_vector = self.user_encoder(browsed_vector)
        predict = torch.stack([self.click_predictor(news_vector, user_vector) for news_vector in candidate_vector], dim=1)
        topic_predict = self.topic_predictor(
            torch.cat((candidate_vector.transpose(0, 1), browsed_vector), dim=1).view(-1, self.args.n_filters)
        )
        topic_true = torch.stack([news['category'] for news in batch['candidate'] + batch['browsed']]).flatten().to(self.args.device)
        class_weight = torch.ones(self.args.n_categories,).to(self.args.device)
        class_weight[0] = 0
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        topic_loss = criterion(topic_predict, topic_true)
        return predict, topic_loss

    def get_user_vector(self, browsed_news_vector):
        return self.user_encoder(browsed_news_vector)

    def get_news_vector(self, candidate):
        return self.news_encoder(candidate)