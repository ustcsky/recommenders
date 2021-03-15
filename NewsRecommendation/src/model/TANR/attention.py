import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(torch.nn.Module):
    def __init__(self, query_vector_dim, input_vector_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(input_vector_dim, query_vector_dim)
        self.query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, input):
        '''
        Args:
            input: batch_size, n_input_vector, input_vector_dim
        Returns:
            result: batch_size, input_vector_dim
        '''
        # batch_size, n_input_vector, query_vector_dim
        tmp = torch.tanh(self.linear(input))
        # batch_size, n_input_vector
        weight = F.softmax(torch.matmul(tmp, self.query_vector), dim=1)
        result = torch.bmm(weight.unsqueeze(dim=1), input).squeeze(dim=1)
        return result