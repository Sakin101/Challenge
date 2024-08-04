import sklearn.metrics

import torch
import torch.nn as nn

import glob, pickle
# from torchinfo import summary
import datetime
import cv2
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch.utils
import torch.utils.data
import torchvision.transforms.v2 as v2

from evaluate import compute_overall_metrics
from paths import MODEL_PATH, PROCESSED_PATH
import sklearn

import timm

import torchvision.models as models
import matplotlib.pyplot as plt

MODEL = 'resnet'

if MODEL == 'resnet':
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    def load_model(filename):
        model_dict = torch.load(filename)
        resnet.load_state_dict(model_dict)

    load_model(MODEL_PATH+'/2024-07-01@09-40-27-resnet-ct-ptinit-accuracy-0.08180712090163934.pth')

    last_dim = resnet.fc.weight.shape[1]
    resnet.fc = torch.nn.Identity()
    
    classifier = torch.nn.Sequential(
                    torch.nn.Linear(in_features=last_dim, out_features=256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.2),
                    torch.nn.Linear(256, out_features=3)
                )
    # total = 6282
    # c1 = 5375
    # c2 = 5806
    # c3 = 5171

    # b1 = (total-c1)/c1
    # b2 = (total-c2)/c2
    # b3 = (total-c3)/c3

    # classifier.bias = torch.nn.Parameter(torch.log(torch.tensor([b1, b2, b3]))) #, requires_grad=False)

    model = resnet
elif MODEL == 'vit':
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    last_layer = list(vit.children())[-1].head
    last_dim = last_layer.weight.shape[1]
    vit.heads = torch.nn.Linear(in_features=last_dim, out_features = 3)

    total = 6282
    c1 = 5375
    c2 = 5806
    c3 = 5171

    b1 = (total-c1)/c1
    b2 = (total-c2)/c2
    b3 = (total-c3)/c3

    vit.heads.bias = torch.nn.Parameter(torch.log(torch.tensor([b1, b2, b3])), requires_grad=False)

    model = vit

    # class ViTMultiLabelClassifier(nn.Module):
    #     def __init__(self, num_classes=3, dropout=0.0, pretrained=True):
    #         super(ViTMultiLabelClassifier, self).__init__()
    #         # Pre-trained ViT as backbone
    #         self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    #         self.vit.head = nn.Identity()

    #         # MLP as classification head
    #         self.classifier = nn.Sequential(
    #             nn.Linear(self.vit.embed_dim, 512),
    #             nn.ReLU(),
    #             nn.Dropout(dropout),
    #             nn.Linear(512, num_classes),
    #         )

    #     def forward(self, x:torch.Tensor)->torch.Tensor:
    #         """ Classify images
    #         Args:
    #         x (torch.Tensor): The input images, of size BxCxHxW.

    #         Returns:
    #         torch.Tensor: a Bx3 vector specifying the confidence for each label (C1,C2,C3 of the CVS)
    #         """
    #         x = self.vit(x)  # Pass input image to ViT
    #         x = self.classifier(x)  # Pass through MLP classifier
    #         return x
    
    # model = ViTMultiLabelClassifier(dropout=0.2)

elif MODEL == 'resnext':
    resnext = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)

    def load_model(filename):
        model_dict = torch.load(filename)
        resnext.load_state_dict(model_dict)

    load_model(MODEL_PATH+'/2024-07-21@17-38-04-resnext-ct-ptinit-accuracy-0.09571781015037593.pth')

    last_dim = resnext.fc.weight.shape[1]
    resnext.fc = torch.nn.Linear(in_features=last_dim, out_features=3)
    total = 6282
    c1 = 5375
    c2 = 5806
    c3 = 5171

    b1 = (total-c1)/c1
    b2 = (total-c2)/c2
    b3 = (total-c3)/c3

    resnext.fc.bias = torch.nn.Parameter(torch.log(torch.tensor([b1, b2, b3]))) #, requires_grad=False)

    model = resnext

NUM_EPOCHS = 50
BATCH_SIZE = 16

with open(PROCESSED_PATH+'/inputs', 'rb') as f:
    inputs = pickle.load(f)

with open(PROCESSED_PATH+'/votes', 'rb') as f:
    outputs = pickle.load(f)

with open(PROCESSED_PATH+'/endo_train_inputs', 'rb') as f:
    endo_inputs = pickle.load(f)

with open(PROCESSED_PATH+'/endo_train_outputs', 'rb') as f:
    endo_outputs = pickle.load(f)

with open(PROCESSED_PATH+'/endo_val_inputs', 'rb') as f:
    endo_val_inputs = pickle.load(f)

with open(PROCESSED_PATH+'/endo_val_outputs', 'rb') as f:
    endo_val_outputs = pickle.load(f)

inputs = torch.tensor(inputs, dtype=torch.uint8)
outputs = torch.tensor(outputs, dtype=torch.float32)/3

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
    # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
)

transform_train = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.2, 1.0)),
    # v2.RandomApply(
    #     [v2.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.8  # not strengthened
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

val_inputs = inputs.view(NUM_OF_VIDS, 18, 3, 224, 224)[:100].view(-1, 3, 224, 224)
val_outputs = outputs.view(NUM_OF_VIDS, 18 * 3)[:100].view(-1, 3)

inputs = inputs.view(NUM_OF_VIDS, 18, 3, 224, 224)[100:]
outputs = outputs.view(NUM_OF_VIDS, 18 * 3)[100:]

print(len(inputs), len(outputs))

def get_train_val_loader(inputs, outputs):
    X_train, X_test, y_train, y_test= train_test_split(inputs, outputs, train_size=0.8, 
                                                    random_state=10, 
                                                    shuffle=True) #, stratify=outputs)

    train_length = len(X_train)
    test_length = len(X_test)

    X_train = X_train.view((train_length*18), 3, 224, 224)
    X_test = X_test.view((test_length*18), 3, 224, 224)

    y_train = y_train.view((train_length*18), 3)
    y_test = y_test.view((test_length*18), 3)

    X_train = torch.concat((X_train, torch.tensor(endo_inputs, dtype=torch.uint8)), dim=0)
    y_train = torch.concat((y_train, torch.tensor(endo_outputs, dtype=torch.float32)), dim=0)

    # X_test = torch.concat((X_test, torch.tensor(endo_val_inputs, dtype=torch.uint8)), dim=0)
    # y_test = torch.concat((y_test, torch.tensor(endo_val_outputs, dtype=torch.float32)), dim=0)

    X_test = torch.concat((X_test, val_inputs), dim=0)
    y_test = torch.concat((y_test, val_outputs), dim=0)

    print(len(X_train), len(X_test))

    y_test[y_test<=0.5] = 0
    y_test[y_test>0.5] = 1

    # # freqs = np.array([1/4210, 1/369, 1/547, 1/249, 1/97, 1/17, 1/317, 1/476])
    # # # freqs = np.sqrt(freqs)
    # # weights = []
    # # for label in y_train:
    # #     num = 4*label[0]+2*label[1]+label[2]
    # #     weights.append(freqs[num])

    # # sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=480, replacement=True)

    # print(len(weights), len(X_train))

    train_dataset = Dataset(X_train, y_train, transform=transform_train)
    test_dataset = Dataset(X_test, y_test, transform=transform_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)#, sampler=sampler)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader

train_dataloader, test_dataloader = get_train_val_loader(inputs, outputs)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    n = 0

    overall_labels = np.zeros((len(train_dataloader.dataset), 3), dtype=np.uint8)
    overall_predictions = np.zeros((len(train_dataloader.dataset), 3))

    for i, (X, y) in enumerate(dataloader):
        overall_labels[n:n+y.size(0)] = (y>0.5).detach().cpu().numpy()
        X, y = X.to(device), y.to(device)
        y_features = resnet(X)
        clustering_loss = 0 #clusteringLoss(y_features, y, loss_fn)
        y_pred = classifier(y_features)
        
        overall_predictions[n:n+y.size(0)] = torch.nn.functional.sigmoid(y_pred).detach().cpu().numpy()

        classifier_loss = loss_fn(torch.nn.functional.sigmoid(y_pred), y)
        loss = classifier_loss + clustering_loss/5

        # print(classifier_loss, clustering_loss)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        n += y.size(0)

    # with open('./img', 'wb') as f:
    #     pickle.dump(X[-1], f)

    assert(n == len(train_dataloader.dataset))
    metrics = compute_overall_metrics(overall_labels, overall_predictions)
    print('train metrics:')
    print(metrics)

    for k in range(3):
        confusion_matrix = sklearn.metrics.confusion_matrix(overall_labels[:,k], overall_predictions[:,k] > 0.5)
        print(confusion_matrix)

    return total_loss/n

def save_model(model, val_map, acc, path=MODEL_PATH, identifier=''):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), 
        os.path.join(
            path,
            f"{datetime.datetime.now().__str__()[:-6].replace(':','-').replace(' ','@')[:-1]}-{identifier}-val-map-{val_map}-acc-{acc}.pth",
        ),
    )


model = model.to(device)
classifier = classifier.to(device)
optimizer = torch.optim.Adam(params=list(model.parameters())+list(classifier.parameters()), lr=2e-4)
loss_fn = torch.nn.MSELoss()

def weightedMSE(predictions, targets):
    weights = targets.clone()
    weights[targets<0.5] = 1
    weights[targets>0.5] = 2
    weights[targets>0.7] = 3

    losses = (predictions-targets)**2*weights / torch.norm(weights, p=1, dim=0).view(1, targets.shape[1]).repeat(targets.shape[0], 1)
    return torch.mean(losses, dim=[0, 1])

def clusteringLoss(features, classes, loss_fn):
    normalized_features = features / torch.norm(features, p=2, dim=1, keepdim=True)
    proximity_matrix = normalized_features @ normalized_features.T

    confusing_examples     = (abs(classes-0.5) < 0.25).float()
    non_confusing_examples = (abs(classes-0.5) > 0.25).float()

    loss = 0

    for i in range(3):
        both_confusing    = confusing_examples[:,i].view(-1, 1)     @ confusing_examples[:,i].view(1, -1)
        neither_confusing = non_confusing_examples[:,i].view(-1, 1) @ non_confusing_examples[:,i].view(1, -1)
        same_cluster      = both_confusing + neither_confusing
        loss += loss_fn(proximity_matrix, same_cluster)

    return loss


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
            y_pred = classifier(model(X))
            
            overall_predictions[j:j+y.size(0)] = torch.nn.functional.sigmoid(y_pred).cpu().numpy()

            j+=y.size(0)
    assert(j == len(test_dataloader.dataset))
    metrics = compute_overall_metrics(overall_labels, overall_predictions)

    print('val metrics:')

    for k in range(3):
        confusion_matrix = sklearn.metrics.confusion_matrix(overall_labels[:,k], overall_predictions[:,k] > 0.5)
        print(confusion_matrix)


    # print(np.max(overall_predictions, axis=0), np.mean(overall_predictions, axis=0), np.std(overall_predictions, axis=0))
    print(f"Epoch {i+1} Training Loss {loss}")
    print(metrics)

    # save_model(model, metrics["mAP"], metrics["accuracy"], MODEL_PATH, 'resnet-60-20-3')

