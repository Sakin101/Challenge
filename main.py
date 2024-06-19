import torch
import glob, pickle
# from torchinfo import summary
import datetime
import cv2
import pandas as pd
import numpy as np
import os

from paths import MODEL_PATH, PROCESSED_PATH

import torchvision.models as models
resnet = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)

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
        if self.transform is not None:
            self,inputs[i], self.outputs[i] = self.transform(self.inputs[i], self.outputs[i])

        return self.inputs[i], self.outputs[i]

def transform(inputs, outputs):
    if np.random.random() < 0.5:
        inputs = torch.flip(inputs, [2])

    # num_of_rotations = np.random.randint(-1, 2)
    # inputs = torch.rot90(inputs, k=num_of_rotations, dims=[1, 2])

    return inputs, outputs

dataset = Dataset(inputs, outputs)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataset.transform=transform
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

layers = list(resnet.children())[:-1]
layers.append(torch.nn.Flatten())
layers.append(torch.nn.Linear(in_features=2048, out_features=3))
model = torch.nn.Sequential(*layers)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(loss.item())

def save_model(model, accuracy, path=MODEL_PATH, identifier=''):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), 
        os.path.join(
            path,
            f"{datetime.datetime.now().__str__()[:-6].replace(':','-').replace(' ','@')[:-1]}-{identifier}-Accuracy-{accuracy}.pth",
        ),
    )


model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters())
loss_fn = torch.nn.BCEWithLogitsLoss()

for i in range(NUM_EPOCHS):
    train_one_epoch(model, train_dataloader, optimizer, loss_fn)

    model.eval()
    correct = torch.zeros((3, )).to(device)
    n = 0
    with torch.no_grad():
        for (X, y) in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_classes = torch.round(torch.nn.functional.sigmoid(y_pred))
            correct += (y == y_classes).sum(dim=0)

            n += y.size(0)


    accuracy = torch.mean(correct)
    print(i, (correct/n).data)
    if i % 5 == 0:
        save_model(model, accuracy, MODEL_PATH, 'resnext-bce')

