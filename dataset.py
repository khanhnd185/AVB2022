import h5py
import os
import numpy as np
from torch.utils.data import Dataset
import torch
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
        self.name = []

        for line in mtl_lines[1:]:
            splitted_line = line.split(',')
            if split != splitted_line[1]:
                continue

            name = splitted_line[0]
            audioname = name[1:-1] + '.wav'
            if split == 'Test':
                y = [0]
            else:
                y = list(map(float, splitted_line[2:]))

            X.append(os.path.join(wav_path, audioname))
            Y.append(y)
            self.name.append(name)

        self.X = np.array(X)
        self.Y = np.array(Y)
    
    def __getitem__(self, i):
        waveform, _ = torchaudio.load(self.X[i])
        return waveform[0], self.Y[i], self.name[i]

    def __len__(self):
        return len(self.X)

class AVBWavType(Dataset):
    def __init__(self, filename, wav_path, split='Train'):
        super(AVBWavType, self).__init__()
        with open(filename) as f:
            mtl_lines = f.read().splitlines()
        X, Y = [], []
        self.name = []

        for line in mtl_lines[1:]:
            splitted_line = line.split(',')
            if split != splitted_line[1]:
                continue

            name = splitted_line[0]
            audioname = name[1:-1] + '.wav'
            if split == 'Test':
                y = [0, 0, 0, 0, 0, 0, 0, 0]
            else:
                y = {
                    "Gasp": [1, 0, 0, 0, 0, 0, 0, 0],
                    "Laugh": [0, 1, 0, 0, 0, 0, 0, 0],
                    "Cry": [0, 0, 1, 0, 0, 0, 0, 0],
                    "Scream": [0, 0, 0, 1, 0, 0, 0, 0],
                    "Grunt": [0, 0, 0, 0, 1, 0, 0, 0],
                    "Groan": [0, 0, 0, 0, 0, 1, 0, 0],
                    "Pant": [0, 0, 0, 0, 0, 0, 1, 0],
                    "Other": [0, 0, 0, 0, 0, 0, 0, 1]
                }[splitted_line[2]]

            X.append(os.path.join(wav_path, audioname))
            Y.append(y)
            self.name.append(name)

        self.X = np.array(X)
        self.Y = np.array(Y)
    
    def __getitem__(self, i):
        waveform, _ = torchaudio.load(self.X[i])
        return waveform[0], self.Y[i], self.name[i]

    def __len__(self):
        return len(self.X)

class AVBH5py(Dataset):
    def __init__(self, filename, wav_path, split='Train'):
        super(AVBH5py, self).__init__()

        with open(filename) as f:
            mtl_lines = f.read().splitlines()
        X, Y = [], []

        for line in mtl_lines[1:]:
            splitted_line = line.split(',')
            if split != splitted_line[1]:
                continue

            name = splitted_line[0]
            audioname = name[1:-1] + '.hdf5'
            y = list(map(float, splitted_line[2:]))

            folder = 'train' if split == 'Train' else 'val'
            X.append(os.path.join(wav_path, folder, audioname))
            Y.append(y)

        self.X = np.array(X)
        self.Y = np.array(Y)
    
    def __getitem__(self, i):
        with h5py.File(self.X[i], 'r') as dataset:
            return torch.Tensor(np.array(dataset['audio'])), self.Y[i]

    def __len__(self):
        return len(self.X)
