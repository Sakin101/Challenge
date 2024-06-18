import torch
import glob, pickle
import cv2
import numpy as np
import os

from paths import MODEL_PATH, PROCESSED_PATH
BATCH_SIZE = 32

import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

resnet = models.resnext101_32x8d()
layers = list(resnet.children())[:-1]
layers.append(torch.nn.Flatten())
layers.append(torch.nn.Linear(in_features=2048, out_features=6))
model = torch.nn.Sequential(*layers)

with open(PROCESSED_PATH+'/inputs', 'rb') as f:
    inputs = pickle.load(f)

with open(PROCESSED_PATH+'/outputs', 'rb') as f:
    outputs = pickle.load(f)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs, self.outputs = inputs, outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.outputs[i]

dataset = Dataset(inputs, outputs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def show_images_labels(ts, ls):
    for t, l in zip(ts, ls):
        t = t/2+0.5
        t = torch.permute(t, (1, 2, 0))
        t = t.cpu().numpy()
        t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
        cv2.imshow('a', t)
        print(l)
        key = cv2.waitKey(0)
        if key == ord('q'):
            return
        elif key == ord('n'):
            continue

def load_model(filename):
    model_dict = torch.load(filename)
    model.load_state_dict(model_dict)


load_model(MODEL_PATH+'/2024-06-18@15-25-37-Accuracy-0.812037037037037.pth')
model = model.to(device)

def find_hard_examples():
    hard_examples=torch.zeros((0, 3, 224, 224)).to(device)
    hard_labels=[]
    model.eval()
    with torch.no_grad():
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            _, y1_pred = torch.max(y_pred[:, :2], dim=1)
            _, y2_pred = torch.max(y_pred[:,2:4], dim=1)
            _, y3_pred = torch.max(y_pred[:,4:], dim=1)

            wrong_c2 = X[y[:,1] != y2_pred]
            hard_examples = torch.cat((hard_examples, wrong_c2))
            hard_labels.extend(y[:,1][y[:,1] != y2_pred].cpu().tolist())
    
    with open('./hard_examples', 'wb') as f:
        pickle.dump(hard_examples, f)

    with open('./hard_labels', 'wb') as f:
        pickle.dump(hard_labels, f)

find_hard_examples()