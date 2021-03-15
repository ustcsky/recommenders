import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, n_heads, input_vector_dim):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.input_vector_dim = input_vector_dim
        self.single_head_dim = input_vector_dim // n_heads
        self.Q = nn.ParameterList([
            nn.Parameter(torch.empty(input_vector_dim, input_vector_dim).uniform_(-0.1, 0.1))
            for _ in range(n_heads)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.empty(input_vector_dim, self.single_head_dim).uniform_(-0.1, 0.1))
            for _ in range(n_heads)
        ])

    def forward(self, input):
        '''
        Args:
            input: batch_size, n_input_vector, input_vector_dim
        Returns:
            result: batch_size, n_input_vector, input_vector_dim
        '''
        result = []
        for i in range(self.n_heads):
            # batch_size, n_input_size, input_vector_dim
            tmp1 = torch.matmul(input, self.Q[i])
            # batch_size, n_input_size, n_input_size
            tmp2 = torch.bmm(tmp1, input.transpose(1, 2))
            weight = F.softmax(tmp2, dim=2)
            # batch_size, n_input_size, input_vector_dim
            weighted = torch.bmm(weight, input)
            result.append(torch.matmul(weighted, self.V[i]))
        return torch.cat(result, dim=2)



