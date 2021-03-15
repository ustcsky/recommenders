import torch
from model.NAML.news_encoder import NewsEncoder
from model.NAML.user_encoder import UserEncoder
from model.NAML.click_predictor import ClickPredictor

def make_model(args, word_embedding):
    return NAML(args, word_embedding)

class NAML(torch.nn.Module):

    def __init__(self, args, word_embedding):
        super(NAML, self).__init__()
        self.args = args
        self.news_encoder = NewsEncoder(args, word_embedding)
        self.user_encoder = UserEncoder(args)
        self.click_predictor = ClickPredictor()

    def forward(self, batch):
        # K+1, batch_size, n_filters
        candidate_vector = torch.stack([self.news_encoder(x) for x in batch['candidate']])
        # batch_size, n_browsed_news, n_filters
        browsed_vector = torch.stack([self.news_encoder(x) for x in batch['browsed']], dim=1)
        # batch_size, n_filters
        user_vector = self.user_encoder(browsed_vector)
        predict = torch.stack([self.click_predictor(news_vector, user_vector) for news_vector in candidate_vector], dim=1)
        return predict

    def get_user_vector(self, browsed_news_vector):
        return self.user_encoder(browsed_news_vector)

    def get_news_vector(self, candidate):
        return self.news_encoder(candidate)