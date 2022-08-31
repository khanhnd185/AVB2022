import torch
import math
import torch.nn as nn
from torch import Tensor
import torchaudio

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
    def __init__(self,
                 input_size:int,
                 num_outs:int,
                 pretrained:bool = False,
                 model_name:str = None):
        super(AUTRANS, self).__init__()
        self.input_size = input_size
        self.seq_len = 40
        self.front_dim = 2048
        self.embed_dim = 1024
        self.nhead = 8
        self.ffdim = 2048
        self.num_outs = num_outs
        
        self.front = nn.Sequential(
            Dense(self.input_size, self.front_dim, activation='relu')
            , Dense(self.front_dim, self.embed_dim, activation='relu')
        )

        self.pos_encoder = PositionalEncoding(self.embed_dim)
        self.encoder_layer1 = nn.TransformerEncoderLayer(self.embed_dim, self.nhead, dim_feedforward=self.ffdim, batch_first=True)
        self.encoder_layer2 = nn.TransformerEncoderLayer(self.embed_dim, self.nhead, dim_feedforward=self.ffdim, batch_first=True)
        self.encoder_layer3 = nn.TransformerEncoderLayer(self.embed_dim, self.nhead, dim_feedforward=self.ffdim, batch_first=True)
        self.encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=6)
        self.encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=2)
        self.encoder3 = nn.TransformerEncoder(self.encoder_layer3, num_layers=2)
        self.avgpool1 = nn.AvgPool1d(2)
        self.avgpool2 = nn.AvgPool1d(2)

        self.end = Dense(self.embed_dim, self.num_outs)

    def forward(self, x:torch.Tensor):
        """
        Args:
            x ((torch.Tensor) - BS x S x 1 x T)
        """

        batch_size, seq_length, t = x.shape
        x = x.view(batch_size*seq_length, 1, t)

        audio_out = self.front(x)
        audio_out = audio_out.view(batch_size, seq_length, -1)
        audio_out = self.pos_encoder(audio_out)

        output = self.encoder1(audio_out)
        #output = torch.transpose(self.avgpool1(torch.transpose(output, 1, 2)), 1, 2)
        #output = self.encoder2(output)
        #output = torch.transpose(self.avgpool2(torch.transpose(output, 1, 2)), 1, 2)
        #output = self.encoder3(output)
        output = self.end(output)

        return output


class AvbWav2vec(nn.Module):
    def __init__(self,
                 bundle,
                 feature:int,
                 num_outs:int,
                 freeze_extractor:bool = True,
                 loss:str = 'ccc',
                 layer:int = -1):
        super(AvbWav2vec, self).__init__()
        self.extractor = bundle.get_model()
        self.layer = layer
        self.linear = nn.Linear(feature, num_outs)
        self.num_outs = num_outs

        if loss == 'ccc':
            self.bn = nn.BatchNorm1d(num_outs)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        else:
            self.bn = nn.Identity()
        self.ac = nn.Sigmoid()

        if freeze_extractor:
            #for p in self.extractor.feature_extractor.parameters():
            for p in self.extractor.parameters():
                p.requires_grad = False

    def forward(self, x, lengths):
        features, lengths = self.extractor.extract_features(x, lengths, self.layer)
        output = features[self.layer - 1].sum(dim=1)
        output = torch.div(output, lengths.unsqueeze(1))
        output = self.linear(output)
        output = self.bn(output)
        output = self.ac(output)

        return output

class AvbWav2vecLstm(nn.Module):
    def __init__(self,
                 bundle,
                 feature:int,
                 num_outs:int,
                 freeze_extractor:bool = True,
                 loss:str = 'ccc',
                 layer:int = 12):
        super(AvbWav2vecLstm, self).__init__()
        self.extractor = bundle.get_model()
        self.rnn = nn.LSTM(feature, 512, num_layers=2, batch_first=True)
        self.linear = nn.Linear(512, num_outs)
        self.layer = layer
        self.num_outs = num_outs
        
        if loss == 'ce':
            self.ac = nn.Identity()
        else:
            self.ac = nn.Sigmoid()
        
        if loss == 'ccc':
            self.bn = nn.BatchNorm1d(num_outs)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        else:
            self.bn = nn.Identity()

        if freeze_extractor:
            #for p in self.extractor.feature_extractor.parameters():
            for p in self.extractor.parameters():
                p.requires_grad = False

    def forward(self, x, lengths):
        features, lengths = self.extractor.extract_features(x, lengths, self.layer)
        output = features[self.layer - 1]
        self.rnn.flatten_parameters()
        output, _ = self.rnn(output)
        last_index = lengths.long() - 1
        output = self.linear(output)
        output = self.bn(torch.transpose(output, 1, 2))
        output = self.ac(torch.transpose(output, 1, 2))
        output = output[range(output.shape[0]), last_index, :]

        return output

class AvbWav2vecFeatureLstm(nn.Module):
    def __init__(self,
                 num_outs:int):
        super(AvbWav2vecFeatureLstm, self).__init__()
        self.rnn = nn.LSTM(768, 512, num_layers=2, batch_first=True)
        self.linear = nn.Linear(512, num_outs)
        self.bn = nn.BatchNorm1d(num_outs)
        self.ac = nn.Sigmoid()

        self.num_outs = num_outs
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x, lengths):
        self.rnn.flatten_parameters()
        output, _ = self.rnn(x)
        last_index = lengths.long() - 1
        output = self.linear(output)
        output = self.bn(torch.transpose(output, 1, 2))
        output = self.ac(torch.transpose(output, 1, 2))
        output = output[range(output.shape[0]), last_index, :]

        return output
