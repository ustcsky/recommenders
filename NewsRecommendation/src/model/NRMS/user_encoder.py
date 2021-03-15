import torch
from model.NRMS.multi_head import MultiHeadSelfAttention
from model.NRMS.additive import AdditiveAttention

class UserEncoder(torch.nn.Module):

    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.multi_head_self_attention = MultiHeadSelfAttention(args.n_heads, args.word_embedding_dim)
        self.additive_attention = AdditiveAttention(args.query_vector_dim, args.word_embedding_dim)

    def forward(self, news_vectors):
        h = self.multi_head_self_attention(news_vectors)
        user_vector = self.additive_attention(h)
        return user_vector
