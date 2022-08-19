import torch
import math
import torch.nn as nn
from torch import Tensor

class  Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='none', bn=False, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        if activation == 'tanh':
            self.ac = nn.Tanh()
        elif activation == 'softmax':
            self.ac = nn.Softmax()
        elif activation == 'sigmoid':
            self.ac = nn.Sigmoid()
        elif activation == 'relu':
            self.ac = nn.ReLU() 
        else:
            self.ac = nn.Identity()
        
        if bn:
            self.bn = nn.BatchNorm1d(out_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        else:
            self.bn = nn.Identity()

        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(1. / out_features))

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.ac(x)
        return x

class MLP(nn.Module):
    def __init__(self, feature_size, num_output):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            Dense(feature_size, 4096, activation='relu')
            ,Dense(4096, 2048, activation='relu')
            ,Dense(2048, 1024, activation='relu')
            ,Dense(1024, 512, activation='relu')
            ,Dense(512, 256, activation='relu')
            ,Dense(256, 128, activation='relu')
            ,Dense(128, 64, activation='relu')
            ,Dense(64, num_output, activation='tanh', bn=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x

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


class AUTRANS(nn.Module):
    def __init__(self, input_size=1600, num_output=10):
        super(AUTRANS, self).__init__()
        self.input_size = input_size
        self.seq_len = 40
        self.front_dim = 2048
        self.embed_dim = 64
        self.nhead = 8
        self.ffdim = 128
        
        self.front = nn.Sequential(
            Dense(input_size, self.front_dim, activation='relu')
            , Dense(self.front_dim, self.embed_dim, activation='relu')
        )

        self.pos_encoder = PositionalEncoding(self.embed_dim)
        self.encoder_layer1 = nn.TransformerEncoderLayer(self.seq_len, self.nhead, dim_feedforward=self.ffdim, batch_first=True)
        self.encoder_layer2 = nn.TransformerEncoderLayer(self.seq_len // 2, self.nhead, dim_feedforward=self.ffdim, batch_first=True)
        self.encoder_layer3 = nn.TransformerEncoderLayer(self.seq_len // 4, self.nhead, dim_feedforward=self.ffdim, batch_first=True)
        self.encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=2)
        self.encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=2)
        self.encoder3 = nn.TransformerEncoder(self.encoder_layer3, num_layers=2)
        self.avgpool1 = nn.AvgPool1d(2)
        self.avgpool2 = nn.AvgPool1d(2)

        self.end = Dense(self.embed_dim, num_output)