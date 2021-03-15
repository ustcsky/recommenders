import torch
import torch.nn as nn

class ClickPredictor(nn.Module):
    def __init__(self):
        super(ClickPredictor, self).__init__()

    def forward(self, news_vector, user_vector):
        predict = torch.bmm(user_vector.unsqueeze(dim=1), news_vector.unsqueeze(dim=2)).flatten()
        return predict