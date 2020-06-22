import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import librosa
import joblib
import glob 


SAMPLE_RATE = 32000
N_MELS = 128
HOP_LENGTH = 347
N_FFT = 128*20
FMIN = 20
FMAX = SAMPLE_RATE//2


def build_spectogram(path, duration):
    y = None
    if duration != 0:
        y, _ = librosa.load(path, sr = SAMPLE_RATE, res_type="kaiser_fast", duration=duration)
    else: 
        y, _ = librosa.load(path, sr = SAMPLE_RATE, res_type="kaiser_fast" )
    M = librosa.feature.melspectrogram(
            y,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH, # 1sec -> 128
            n_fft=N_FFT,
            fmin=FMIN,
            fmax=FMAX,
        ).astype(np.float32)
    #joblib.dump(M, "./input/mel_pickles/{}.pkl".format(path.split("/")[-1][:-4]))
    return M

"""
if __name__ == "__main__":
    files = glob.glob("./input/train_audio/*.mp3")
    train_df = pd.read_csv("./input/train.csv")
    for i in range(len(train_df)):
        if ((i+1)%100):
            print("Finish",)
        row = train_df.iloc[i]
        duration = np.min([400, row["duration"]])
        if duration != row["duration"]:
            print("truncated audio")

        fp = base_path + "train_audio/" + row["ebird_code"] +"/"+ row["filename"]

        build_spectogram(fp,)
"""