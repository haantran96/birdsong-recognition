#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=0
export EPOCHS="100"
export TRAIN_BATCH_SIZE=64
export TEST_BATCH_SIZE=10
export BASE_MODEL="resnet34"
export TRAINING_FOLDS_CSV="../input/train_folds.csv"
export CROPS="512"

export TRAINING_FOLDS="(0,1,2,3)"
export VALIDATION_FOLDS="(4,)"
python3 train.py

export TRAINING_FOLDS="(0,1,2,4)"
export VALIDATION_FOLDS="(3,)"
python3 train.py

export TRAINING_FOLDS="(0,1,4,3)"
export VALIDATION_FOLDS="(2,)"
python3 train.py

export TRAINING_FOLDS="(0,4,2,3)"
export VALIDATION_FOLDS="(1,)"
python3 train.py

export TRAINING_FOLDS="(4,1,2,3)"
export VALIDATION_FOLDS="(0,)"
python3 train.py




