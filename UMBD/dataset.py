import torch
import cv2
import numpy as np
from tqdm import tqdm
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
import pandas as pd
import os

import glob
import cv2
import numpy as np



class RetinopathyDatasetTrain(Dataset):
    def __init__(self, csv_file, transform=None, split=(-1,-1), test=False):
        
        self.data = np.load(csv_file, allow_pickle=True).item()
        self.transform = transform
        self.split = 0 if split[0] < 0 else split[0]
        self.total = 1 if split[1] < 0 else split[1]
        self.start = int(len(self.data) // self.total) * self.split
        self.test = test
        
    def __len__(self):
        return int(len(self.data['target']) // self.total) - 1


    def __getitem__(self, idx):

        im = torch.tensor(self.data['data'][self.start + idx])
        label = torch.tensor(np.argmax(self.data['target'][self.start + idx]))

        return im, label

