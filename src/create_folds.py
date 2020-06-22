import pandas as pd 
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

if __name__ == "__main__":
    df = pd.read_csv("./input/train.csv")
    corrupted_files = [
    'XC195038.mp3',
    ]

    df = df[~df["filename"].isin(corrupted_files)]

    df.loc[:, "kfold"] = 1
    df = df.sample(frac=1).reset_index(drop=True)
    df["ebird_code"] = df["ebird_code"].astype("category")
    df["ebird_lbl"] = df["ebird_code"].cat.codes
    X = df.filename.values
    y = df.ebird_lbl.values

    print(df.dtypes)

    mskf = StratifiedKFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(mskf.split(X,y)):
        print("TRAIN:", trn_, "VAL:", val_)
        df.loc[val_, "kfold"] = fold

    print(df.kfold.value_counts())
    df.to_csv("./input/train_folds.csv", index = False)

