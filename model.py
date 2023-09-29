import math

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models 


class BehEncoder(nn.Module):
    def __init__(self, code_length, n_frame, dropout=0.1):
        super(BehEncoder, self).__init__()
        self.code_length = code_length
        self.n_frame = n_frame

        resnet = models.resnet18(True)
        resnet = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*resnet)

        self.dropout = nn.Dropout(dropout)

        in_channel = 512
        self.in_channel = in_channel
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.in_channel, nhead=8, dim_feedforward=self.in_channel, dropout=0.1),
            num_layers=1
        )
        self.pos_encoder = PositionalEncoding(self.in_channel, dropout=dropout)
        self.relu = nn.LeakyReLU(0.1)

        # reconstruction
        self.pos_embedding = nn.Embedding(n_frame, in_channel)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=in_channel, nhead=8, dim_feedforward=in_channel, dropout=0.1),
            num_layers=1
        )
        self.reconstruct_layer = nn.Linear(in_channel, in_channel)

    def forward(self, x, mask_ratio=0.0):
        batch_size, n_frame, c, h, w = x.size()
        x = self.backbone(x.view(batch_size * n_frame, c, h, w)).view(batch_size, n_frame, -1)  # (batch_size * n_frame * feature)

        if mask_ratio != 0:
            # masked x for training
            x, y, mask = self._random_mask(x, mask_ratio)
        else:
            y = x.detach()
            mask = torch.ones((batch_size, n_frame), dtype=torch.int, device=x.device)

        x = x.transpose(0, 1)  # (batch_size * n_frame * feature) -> (n_frame * batch_size * feature)
        x = self.pos_encoder(x * math.sqrt(self.in_channel))  # dropout in pos_encoder
        x = self.transformer_encoder(x)  # (n_frame * batch_size * feature)

        beh_embedding = self.dropout(self.relu(x)).mean(0)  # (batch_size * feature)
        x = self.pos_embedding(
            torch.arange(0, self.n_frame).repeat((batch_size, 1)).to(beh_embedding.device)).transpose(0, 1)
        x = self.transformer_decoder(x, beh_embedding.unsqueeze(0)).transpose(0, 1)
        x = self.reconstruct_layer(x)

        return beh_embedding, x, y, mask

    def _random_mask(self, x, p=0.3):
        y = x.detach()
        prob = torch.rand((x.size(0), x.size(1)))
        mask = torch.tensor((prob < p), dtype=torch.int, device=x.device)
        fill = (prob / p) < 0.1
        fill_val_idx = (torch.randint(0, x.size(0), (fill.sum(),)), torch.randint(0, x.size(1), (fill.sum(),)))

        # mask out
        x = x * (1 - mask).unsqueeze(2)
        # random fill
        x[fill] = y[fill_val_idx]
        return x, y, mask
    

class PhaseEncoder(nn.Module):
    def __init__(self, code_length, n_frame, dropout=0.3):
        super(PhaseEncoder, self).__init__()
        self.code_length = code_length
        self.n_frame = n_frame

        resnet = models.resnet18(True)
        resnet = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*resnet)

        self.dropout = nn.Dropout(dropout)

        in_channel = 512
        self.in_channel = in_channel
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.in_channel, nhead=8, dim_feedforward=self.in_channel, dropout=0.2),
            num_layers=1
        )
        self.pos_encoder = PositionalEncoding(self.in_channel, dropout=dropout)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        batch_size, n_frame, c, h, w = x.size()
        x = self.backbone(x.view(batch_size * n_frame, c, h, w)).view(batch_size, n_frame, -1)  # (batch_size * n_frame * feature)

        x = x.transpose(0, 1)  # (batch_size * n_frame * feature) -> (n_frame * batch_size * feature)
        x = self.pos_encoder(x * math.sqrt(self.in_channel))  # dropout in pos_encoder
        x = self.transformer_encoder(x)  # (n_frame * batch_size * feature)

        x = self.dropout(self.relu(x)).mean(0)  # (batch_size * feature)
        return x


class SPHashNet(nn.Module):
    def __init__(self, code_length, n_frame):
        super(SPHashNet, self).__init__()

        in_channel = 512
        self.beh_encoder = BehEncoder(code_length, n_frame)
        self.phase_encoder = PhaseEncoder(code_length, n_frame)

        self.h_trans_layer = nn.Sequential(
            nn.Linear(in_channel, in_channel // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(in_channel // 2, code_length)
        )
        self.bn = nn.BatchNorm1d(code_length)
        self.tanh = nn.Tanh()

        self.l_trans_layer = nn.Sequential(
            nn.Linear(in_channel * 2, in_channel // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(in_channel // 2, code_length)
        )

    def forward(self, x, mask_ratio=0.0):
        batch_size = x.size(0)
        x_p = self.phase_encoder(x)
        x_b, x_rec, y_rec, mask = self.beh_encoder(x, mask_ratio)

        g_h = self.h_trans_layer(x_p)  # (batch_size * k)
        if batch_size == 1:
            g_h = self.tanh(g_h)
        else:
            g_h = self.tanh(self.bn(g_h))

        x = torch.cat([x_p, x_b.detach()], dim=1)
        g_l = self.tanh(self.l_trans_layer(x))
        g = g_h * g_l

        return g, g_h, x_b, x_rec, y_rec, mask


class HashCenter:
    def __init__(self, n_cls, code_length):
        self.p = torch.rand(n_cls, code_length)
        self.centers = None

    def get_centers(self, device='cpu'):
        centers = self.p.clone()
        centers[centers < 0.5] = -1
        centers[centers >= 0.5] = 1
        self.centers = torch.tensor(centers, dtype=torch.float, device=device, requires_grad=True)
        return self.centers

    def state_dict(self):
        return {"p": self.p.clone()}

    def load_state_dict(self, state_dict):
        self.p = state_dict["p"]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)