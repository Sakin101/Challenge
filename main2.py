import torch
import glob, pickle
# from torchinfo import summary
import datetime
import cv2
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as v2

from evaluate import compute_overall_metrics
from paths import MODEL_PATH, PROCESSED_PATH

import torchvision.models as models

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

def load_model(filename):
    model_dict = torch.load(filename)
    resnet.load_state_dict(model_dict)

load_model(MODEL_PATH+'/2024-06-25@20-08-00-resnet-ct-ptinit-loss-0.025519938849401277.pth')

last_dim = resnet.fc.weight.shape[1]
resnet.fc = torch.nn.Linear(in_features=last_dim, out_features=3)

model = resnet


# vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
# last_layer = list(vit.children())[-1].head
# last_dim = last_layer.weight.shape[1]
# vit.heads = torch.nn.Linear(in_features=last_dim, out_features = 3)

# model = vit

NUM_EPOCHS = 40
BATCH_SIZE = 32

with open(PROCESSED_PATH+'/inputs', 'rb') as f:
    inputs = pickle.load(f)

with open(PROCESSED_PATH+'/outputs', 'rb') as f:
    outputs = pickle.load(f)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs, transform=None):
        self.inputs, self.outputs = inputs, outputs
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        x = torch.tensor(self.inputs[i], dtype=torch.float32)/255
        if self.transform is not None:
            x = self.transform(x)

        return x, self.outputs[i]

normalize = v2.Normalize(
    mean=[0.3736, 0.2172, 0.2071], std=[0.2576, 0.20095, 0.1949]
)

transform_train = v2.Compose(
    [
    v2.RandomHorizontalFlip(),
    normalize
    ]
)

transform_test = v2.Compose(
    [
    normalize
    ]
)

def get_train_val_loader(inputs, outputs):
    X_train, X_test, y_train, y_test= train_test_split(inputs, outputs, train_size=0.8, 
                                                    random_state=None, 
                                                    shuffle=True, stratify=outputs)

    print("Class distribution of train set")
    print(y_train.sum(dim=0), y_train.size(0))
    print()
    print("Class distribution of test set")
    print(y_test.sum(dim=0), y_test.size(0))

    train_dataset = Dataset(X_train, y_train, transform=transform_train)
    test_dataset = Dataset(X_test, y_test, transform=transform_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader

train_dataloader, test_dataloader = get_train_val_loader(inputs, outputs)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    n = 0
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += y.size(0)

    return total_loss/n

def save_model(model, val_map, path=MODEL_PATH, identifier=''):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), 
        os.path.join(
            path,
            f"{datetime.datetime.now().__str__()[:-6].replace(':','-').replace(' ','@')[:-1]}-{identifier}-val-map-{val_map}.pth",
        ),
    )


model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters())
loss_fn = torch.nn.BCEWithLogitsLoss()


overall_labels = np.zeros((350*18, 3), dtype=np.uint8)
overall_predictions = np.zeros((350*18, 3))

for i in range(NUM_EPOCHS):
    loss = train_one_epoch(model, train_dataloader, optimizer, loss_fn)

    model.eval()

    j = 0
    with torch.no_grad():
        for (X, y) in test_dataloader:
            overall_labels[j:j+y.size(0)] = y.numpy()
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            
            overall_predictions[j:j+y.size(0)] = torch.nn.functional.sigmoid(y_pred).cpu().numpy()

            j+=y.size(0)

    metrics = compute_overall_metrics(overall_labels, overall_predictions)
    print(f"Epoch {i+1} Training Loss {loss}")
    print(metrics)

    # if i % 5 == 0:
    #     save_model(model, metrics["mAP"], MODEL_PATH, 'resnet-base')

