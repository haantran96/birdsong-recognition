import numpy as np 
import pandas as pd 
import librosa
import joblib
import glob 
import random

SAMPLE_RATE = 32000
N_MELS = 128
HOP_LENGTH = 347
N_FFT = 128*20
FMIN = 20
FMAX = SAMPLE_RATE//2


def freq_mask(spec, F=30, replace_with_zero=True):
    num_mel_channels = spec.shape[0]
    f = random.randrange(0, F)
    f_zero = random.randrange(0, num_mel_channels - f)

    # avoids randrange error if values are equal and range is empty
    if (f_zero == f_zero + f): return spec

    mask_end = random.randrange(f_zero, f_zero + f) 
    if (replace_with_zero): 
        spec[f_zero:mask_end] = 0
    else: 
        spec[f_zero:mask_end] = spec.mean()
    return spec

def do_random_crop(img,crop):
    """https://github.com/OsciiArt/Freesound-Audio-Tagging-2019/blob/master/src/utils.py"""
    img_new = np.zeros([img.shape[0], crop], np.float32)
    if img.shape[1] < crop:
        shift = np.random.randint(0, crop - img.shape[1])
        img_new[:, shift:shift + img.shape[1]] = img
    elif img.shape[1] == crop:
        img_new = img
    else:
        shift = np.random.randint(0, img.shape[1] - crop)
        img_new = img[:, shift:shift + crop]
    return img_new


def build_spectogram(path):
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