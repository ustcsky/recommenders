import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
class FM_Layer(nn.Module):
    def __init__(self, n=10, k=5):
        super(FM_Layer, self).__init__()
        self.n = n
        self.k = k
        self.linear = nn.Linear(self.n, 1)   # 前两项线性层
        self.V = nn.Parameter(torch.randn(self.n, self.k))   # 交互矩阵
        nn.init.uniform_(self.V, -0.1, 0.1)
    def fm_layer(self, x):
        linear_part = self.linear(x)
        # print('linear_part.shape:', linear_part.shape)
        # print('x.shape:', x.shape)
        # print('self.V.shape:', self.V.shape)
        interaction_part_1 = torch.matmul(x, self.V)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        # print('interaction_part_1.shape:', interaction_part_1.shape)
        interaction_part_2 = torch.matmul(torch.pow(x, 2), torch.pow(self.V, 2))
        # print('interaction_part_2.shape:', interaction_part_2.shape)
        tmp = torch.sum(interaction_part_2 - interaction_part_1, -1, keepdim=True)
        # print('tmp.shape:',tmp.shape)
        output = linear_part + 0.5 * tmp
        return output
    def forward(self, x):
        return self.fm_layer(x)

fm = FM_Layer(6, 1)
x = torch.randn(400, 6)
output = fm(x) # (400, 1)

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