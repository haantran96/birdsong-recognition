import numpy as np
import torch
import pandas as pd 
import joblib
from PIL import Image
import librosa
from utils import build_spectogram
import audiomentations


BASE_DIR = "./input/train_audio/"

class BirdDatasetTrain:
    def __init__(self, folds, mean, std):
        df = pd.read_csv("./input/train_folds.csv")

        df = df[["filename","ebird_code", "ebird_lbl", "duration", "kfold"]]
        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.filenames = df.filename.values
        self.ebird_lbls = df.ebird_lbl.values
        self.ebird_codes = df.ebird_code.values
        self.durations = df.duration.values
        self.aug = None   

    def __len__(self):
        return (len(self.filenames))

    
    def __getitem__ (self,item):
        fp = BASE_DIR + self.ebird_codes[item] + "/" +self.filenames[item]
        duration = self.durations[item]
        mel_spec = build_spectogram(fp,duration)
        print (mel_spec)
        return {
            "audio": torch.tensor(mel_spec, dtype=torch.float),
            "ebird_lbl":torch.tensor(self.ebird_lbls[item], dtype=torch.long)
        }
        