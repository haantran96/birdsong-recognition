import numpy as np
import torch
import pandas as pd 
import joblib
from PIL import Image
import librosa
from utils import *
#import audiomentations


BASE_DIR = "./input/train_audio/"

class BirdDatasetTrain:
    def __init__(self, folds,freq_mask=False, crop = 512):
        df = pd.read_csv("./input/train_folds.csv")

        df = df[["filename","ebird_code", "ebird_lbl", "duration", "kfold"]]
        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.filenames = df.filename.values
        self.ebird_lbls = df.ebird_lbl.values
        self.ebird_codes = df.ebird_code.values
        self.freq_mask = freq_mask
        self.crop = crop 
    
    def __len__(self):
        return (len(self.filenames))

    
    def __getitem__ (self,item):
        fp = BASE_DIR + self.ebird_codes[item] + "/" +self.filenames[item]
        mel_spec = build_spectogram(fp)
        mel_spec = do_random_crop(mel_spec, self.crop)

        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std()+1e-7)
        
        if self.freq_mask:
            mel_spec = freq_mask(mel_spec)
        
        mel_spec = mel_spec.reshape([1,mel_spec.shape[0],mel_spec.shape[1]])
        return {
            "audio": torch.tensor(mel_spec,dtype=torch.float),
            "ebird_lbl":torch.tensor(self.ebird_lbls[item], dtype=torch.long)
        }
        