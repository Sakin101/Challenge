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

MODEL = 'resnet'

if MODEL == 'resnet':
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    def load_model(filename):
        model_dict = torch.load(filename)
        resnet.load_state_dict(model_dict)

    load_model(MODEL_PATH+'/2024-06-26@22-34-14-resnet-ct-ptinit-loss-0.009877337941753805.pth')

    last_dim = resnet.fc.weight.shape[1]
    resnet.fc = torch.nn.Linear(in_features=last_dim, out_features=3)

    model = resnet
else:
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
    last_layer = list(vit.children())[-1].head
    last_dim = last_layer.weight.shape[1]
    vit.heads = torch.nn.Linear(in_features=last_dim, out_features = 3)

    model = vit

NUM_EPOCHS = 60
BATCH_SIZE = 32

with open(PROCESSED_PATH+'/inputs', 'rb') as f:
    inputs = pickle.load(f)

with open(PROCESSED_PATH+'/outputs', 'rb') as f:
    outputs = pickle.load(f)

inputs = torch.tensor(inputs, dtype=torch.uint8)
outputs = torch.tensor(outputs, dtype=torch.uint8)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs, transform=None):
        self.inputs, self.outputs = inputs, outputs
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        x = self.inputs[i]
        if self.transform is not None:
            x = self.transform(x)

        return x, self.outputs[i]

normalize = v2.Normalize(
    mean=[0.3736, 0.2172, 0.2071], std=[0.2576, 0.20095, 0.1949]
)

transform_train = v2.Compose([
    # v2.RandomResizedCrop(224, scale=(0.2, 1.0)),
    # v2.RandomApply(
    #     [v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
    # ),
    # v2.RandomGrayscale(p=0.2),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    normalize,
])

transform_test = v2.Compose(
    [
    v2.ToDtype(dtype=torch.float32, scale=True),
    normalize
    ]
)

NUM_OF_VIDS = len(inputs)//18

print(NUM_OF_VIDS)

inputs = inputs.view(NUM_OF_VIDS, 18, 3, 224, 224)
outputs = outputs.view(NUM_OF_VIDS, 18 * 3)

def get_train_val_loader(inputs, outputs):
    X_train, X_test, y_train, y_test= train_test_split(inputs, outputs, train_size=0.8, 
                                                    random_state=1, 
                                                    shuffle=True) #, stratify=outputs)

    train_length = len(X_train)
    test_length = len(X_test)

    X_train = X_train.view((train_length*18), 3, 224, 224)
    X_test = X_test.view((test_length*18), 3, 224, 224)

    y_train = y_train.view((train_length*18), 3)
    y_test = y_test.view((test_length*18), 3)

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


overall_labels = np.zeros((len(test_dataloader.dataset), 3), dtype=np.uint8)
overall_predictions = np.zeros((len(test_dataloader.dataset), 3))

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
    assert(j == len(test_dataloader.dataset))
    metrics = compute_overall_metrics(overall_labels, overall_predictions)
    print(f"Epoch {i+1} Training Loss {loss}")
    print(metrics)

    if i % 5 == 0:
        save_model(model, metrics["mAP"], MODEL_PATH, 'resnet-60-9-3')

