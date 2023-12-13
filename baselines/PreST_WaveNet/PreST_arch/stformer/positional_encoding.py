import torch
from torch import nn
import math

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 4000, minute_size=288, weekday_size=7):
        super().__init__()
        # minute_size代表一天中有多少个时间步，这里一个时间步是5分钟，所以一天有12*24=288个时间步
        self.tod_embedding = FixedEmbedding(minute_size, hidden_dim)
        self.dow_embedding = FixedEmbedding(weekday_size, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1),requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1),requires_grad=True)
        self.cita = nn.Parameter(torch.ones(1),requires_grad=True)

    def forward(self, input_data, tod=None, dow=None, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N*T, C'].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        B, NT, C = input_data.shape
        # input_data = input_data.view(batch_size*num_nodes, num_patches, num_feat)
        # 时域周期性映射
        te = None
        if (tod is not None) & (dow is not None):
            tod = (tod.squeeze(-1)).long()
            dow = (dow.squeeze(-1)).long()
            te_tod = self.tod_embedding(tod)
            te_dow = self.dow_embedding(dow)
            te = te_tod + self.alpha*te_dow

        # positional encoding
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
            if te != None:
                te = te
        else:
            pe = self.position_embedding[index].unsqueeze(0)
            if te != None:
                te = te[:, index, :]

        input_data = input_data + self.beta*pe
        if te != None:
            input_data = input_data + self.cita*te

        input_data = self.dropout(input_data)
        # reshape
        # input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        return input_data
