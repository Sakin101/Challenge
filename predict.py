import glob
import json

import torch
import torchvision.models as models
import torchvision.transforms.v2 as v2

import numpy as np

import pickle

from paths import TEST_DATA_PATH, TEST_PROCESSED_PATH, MODEL_PATH

BATCH_SIZE=32
NUM_OF_VIDS=100
device='cpu'

videos = sorted(glob.glob(TEST_DATA_PATH + '/videos/*'))

vid_indices = []
for video in videos:
    print(video)
    vid_indices.append(video[-40:-4])

resnet = models.resnet50()

def load_model(model, filename):
    model_dict = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(model_dict)
    return model
    

last_dim = resnet.fc.weight.shape[1]
resnet.fc = torch.nn.Linear(in_features=last_dim, out_features=3)

model = resnet
model = load_model(model, MODEL_PATH+'/2024-07-12@14-59-17-resnet-60-20-3-val-map-0.5941339087377404-acc-0.6061728395061728.pth')

model = model.to(device)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, transform):
        self.inputs = inputs
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        x = self.inputs[i]
        if self.transform is not None:
            x = self.transform(x)

        return x
    
with open(TEST_PROCESSED_PATH+'/inputs', 'rb') as f:
    inputs = pickle.load(f)

inputs = torch.tensor(inputs, dtype=torch.uint8)
print(len(inputs))

normalize = v2.Normalize(
    mean=[0.3736, 0.2172, 0.2071], std=[0.2576, 0.20095, 0.1949]
)

transform_test = v2.Compose(
    [
    v2.ToDtype(dtype=torch.float32, scale=True),
    normalize
    ]
)

dataset = Dataset(inputs, transform=transform_test)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

overall_predictions = torch.zeros((len(dataloader.dataset), 3)).to(device)

model.eval()

j = 0
with torch.no_grad():
    for X in dataloader:
        X = X.to(device)
        y_pred = model(X)
        # print(y_pred)
        overall_predictions[j:j+X.size(0)] = torch.nn.functional.sigmoid(y_pred)

        j+=X.size(0)
assert(j == len(dataloader.dataset))

overall_predictions = overall_predictions.view(NUM_OF_VIDS, 18, 3)

pred_dict = {}

for vid_index, pred in zip(vid_indices, overall_predictions):
    pred_dict[vid_index] = {}
    for i, j in zip(range(0, 30*90, 150), range(18)):
        pred_dict[vid_index][i] = pred[j].cpu().tolist()
    
with open("predictions.json", "w") as outfile: 
    json.dump(pred_dict, outfile)

