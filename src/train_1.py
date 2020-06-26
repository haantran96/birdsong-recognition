import os
import ast
from dispatcher import MODEL_DISPATCHER
from dataset import BirdDatasetTrain
import torch
import torch.nn as nn
from tqdm import tqdm

DEVICE = "cuda"

TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
EPOCHS = int(os.environ.get("EPOCHS"))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL = os.environ.get("BASE_MODEL")

CROPS = int(os.environ.get("CROPS"))


def loss_fn(output,label):
    l1 = nn.BCELoss()(output,label)
    return l1


def train(dataset, data_loader, model, optimizer):
    model.train()

    for bi, d in enumerate(data_loader):        
        audio = d["audio"]
        label = d["ebird_lbl"]

        audio = audio.to(DEVICE, dtype=torch.float)
        label = label.to(DEVICE, dtype = torch.float)
        optimizer.zero_grad()
        output = model(audio)
        proba = output["multilabel_proba"]

        loss = loss_fn(proba, label)

        loss.backward()
        optimizer.step()
        if (bi % 100 == 0):
          print("FINISH (TRAIN) {} / {}, loss: {}".format(bi,len(dataset),loss.item()))


def evaluate(dataset, data_loader, model):
    model.eval()
    final_loss = 0
    counter = 0
    for bi, d in enumerate(data_loader):
        counter += 1        
        audio = d["audio"]
        label = d["ebird_lbl"]

        audio = audio.to(DEVICE, dtype=torch.float)
        label = label.to(DEVICE, dtype = torch.float)
        
        outputs = model(audio)
        proba = outputs["multilabel_proba"]
        loss = loss_fn(outputs, proba)
        final_loss += loss.item()
        
    return final_loss / counter
       

       

def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)

    train_dataset = BirdDatasetTrain(
        folds=TRAINING_FOLDS,
        freq_mask=False,
        crop=CROPS,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    valid_dataset = BirdDatasetTrain(
        folds=VALIDATION_FOLDS,
        freq_mask=False,
        crop=CROPS,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.3, verbose=True)
    """
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    """
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}/{TRAINING_FOLDS}")
        train(train_dataset, train_loader, model, optimizer)
        val_score = evaluate(valid_dataset, valid_loader, model)
        print("VAL SCORE", val_score)
        scheduler.step(val_score)
        torch.save(model.state_dict(), "{}_fold_{}.bin".format(BASE_MODEL,VALIDATION_FOLDS[0]))


if __name__ == "__main__":
    main()

