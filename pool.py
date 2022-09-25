import torch
import torch.nn as nn

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, x, leng):
        ret, _ =  torch.max(x, dim=1)
        return ret

class LastPooling(nn.Module):
    def __init__(self):
        super(LastPooling, self).__init__()

    def forward(self, x, leng):
        last_index = leng.long() - 1
        ret = x[range(x.shape[0]), last_index, :]
        return ret

class FirstPooling(nn.Module):
    def __init__(self):
        super(FirstPooling, self).__init__()

    def forward(self, x, leng):
        ret = x[range(x.shape[0]), 0, :]
        return ret

class MaskAvgPooling(nn.Module):
    def __init__(self):
        super(MaskAvgPooling, self).__init__()

    def forward(self, x, leng):
        ret, _ =  torch.sum(x, dim=1)
        ret = torch.div(ret, leng.unsqueeze(1))
        return ret

class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, x, leng):
        ret =  torch.mean(x, dim=1)
        return ret

class PoolingWrapper(nn.Module):
    def __init__(self, name):
        super(PoolingWrapper, self).__init__()
        self.pool = {
            "max": MaxPooling(),
            "mean": AvgPooling(),
            "last": LastPooling(),
            "first": FirstPooling(),
            "maskavg": MaskAvgPooling(),
        }[name]

    def forward(self, x, leng):
        return self.pool(x, leng)