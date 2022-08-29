import os
import numpy as np
from torch.utils.data import Dataset
import torchaudio

class AVBFeature(Dataset):
    def __init__(self, filename, feature_dict, split='Train'):
        super(AVBFeature, self).__init__()
        with open(filename) as f:
            mtl_lines = f.read().splitlines()
        X, Y = [], []

        for line in mtl_lines[1:]:
            splitted_line = line.split(',')
            if split != splitted_line[1]:
                continue

            name = splitted_line[0]
            audioname = name[1:-1] + '.wav'
            y = list(map(float, splitted_line[2:]))

            if audioname in feature_dict:
                X.append(feature_dict[audioname])
                Y.append(y)

        self.X = np.array(X)
        self.Y = np.array(Y)
    
    def __getitem__(self, i):
        return self.X[i] , self.Y[i]

    def __len__(self):
        return len(self.X)

class AVBWav(Dataset):
    def __init__(self, filename, wav_path, split='Train'):
        super(AVBWav, self).__init__()
        with open(filename) as f:
            mtl_lines = f.read().splitlines()
        X, Y = [], []

        for line in mtl_lines[1:]:
            splitted_line = line.split(',')
            if split != splitted_line[1]:
                continue

            name = splitted_line[0]
            audioname = name[1:-1] + '.wav'
            y = list(map(float, splitted_line[2:]))

            X.append(os.path.join(wav_path, audioname))
            Y.append(y)

        self.X = np.array(X)
        self.Y = np.array(Y)
    
    def __getitem__(self, i):
        waveform, _ = torchaudio.load(self.X[i])
        return waveform[0], self.Y[i]

    def __len__(self):
        return len(self.X)
