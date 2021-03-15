import torch
from model.NAML.attention import Attention

class UserEncoder(torch.nn.Module):

    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.browsed_news_attention = Attention(args.query_vector_dim, args.n_filters)

    def forward(self, news_vectors):
        user_vector = self.browsed_news_attention(news_vectors)
        return user_vector
