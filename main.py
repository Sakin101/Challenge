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

def show_image(t):
    t = t/2+0.5
    t = torch.permute(t, (1, 2, 0))
    t = t.numpy()
    t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    cv2.imshow('a', t)
    if cv2.waitKey(0) == ord('q'):
        return

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs, self.outputs = inputs, outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.outputs[i]

dataset = Dataset(inputs, outputs)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

layers = list(resnet.children())[:-1]
layers.append(torch.nn.Flatten())
layers.append(torch.nn.Linear(in_features=2048, out_features=6))
model = torch.nn.Sequential(*layers)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred[:,:2], y[:,0]) + loss_fn(y_pred[:,2:4], y[:,1]) + loss_fn(y_pred[:,4:6], y[:,2])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(loss.item())

def save_model(model, accuracy, path=MODEL_PATH):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), 
        os.path.join(
            path,
            f"{datetime.datetime.now().__str__()[:-6].replace(':','-').replace(' ','@')[:-1]}-Accuracy-{accuracy}.pth",
        ),
    )


model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

for i in range(NUM_EPOCHS):
    train_one_epoch(model, train_dataloader, optimizer, loss_fn)

    model.eval()
    n = c1 = c2 = c3 = 0
    with torch.no_grad():
        for (X, y) in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            _, y1_pred = torch.max(y_pred[:, :2], dim=1)
            _, y2_pred = torch.max(y_pred[:,2:4], dim=1)
            _, y3_pred = torch.max(y_pred[:,4:], dim=1)
            c1 += (y[:,0] == y1_pred).sum().item()
            c2 += (y[:,1] == y2_pred).sum().item()
            c3 += (y[:,2] == y3_pred).sum().item()
            n += y.size(0)


    accuracy = (c1/n+c2/n+c3/n)/3
    print(i, c1/n, c2/n, c3/n)
    if i % 5 == 0:
        save_model(model, accuracy, MODEL_PATH)

