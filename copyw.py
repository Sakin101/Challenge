import os
import datetime
from paths import MODEL_PATH

import torch
import pickle

import torchvision.transforms.v2 as v2

import moco.loader
import moco.builder

import torch.nn as nn
import numpy as np


BATCH_SIZE=256
NUM_EPOCHS=20
GROUP_SIZE=3

with open('./videos/inputs', 'rb') as f:
    inputs = (pickle.load(f))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, group_size):
        self.inputs = torch.tensor(inputs, dtype=torch.uint8)
        self.transform = transform
        self.group_size = group_size
        assert(90*5%group_size == 0)

    def __len__(self):
        return len(self.inputs)//self.group_size

    def __getitem__(self, i):
        key   = self.inputs[i//self.group_size + np.random.randint(0, self.group_size)]
        value = self.inputs[i//self.group_size + np.random.randint(0, self.group_size)]

        return self.transform(key), self.transform(value)

normalize = v2.Normalize(
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean=[0.3736, 0.2172, 0.2071], std=[0.2576, 0.20095, 0.1949]
)
# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
augmentation = [
    v2.RandomResizedCrop(224, scale=(0.2, 1.0)),
    v2.RandomApply(
        [v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
    ),
    v2.RandomGrayscale(p=0.2),
    v2.ToPILImage(),
    v2.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    normalize,
]

train_dataset = Dataset(transform=v2.Compose(augmentation), group_size=GROUP_SIZE)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

import torchvision.models as models

key_dim=128
dictionary_size=8192
device='cuda'

MODEL = 'resnet'

if MODEL == 'resnet':
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    def load_model(model, filename):
        model_dict = torch.load(filename, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict)
        return model

    # resnet = load_model(resnet, MODEL_PATH+'/2024-06-26@16-01-57-resnet-ct-ptinit-loss-0.015230147587898814.pth')

    for name, param in resnet.named_parameters():
        if 'bn' in name:
            param.requires_grad = False

    out_dim = resnet.fc.weight.shape[0]

    encoder = torch.nn.Sequential(
        resnet,
        nn.ReLU(),
        nn.Linear(in_features=out_dim, out_features=key_dim)
    )

    encoder = load_model(encoder, MODEL_PATH+'/2024-06-27@12-07-00-resnet+mlp-ct-accuracy-0.1293545081967213.pth')

elif MODEL == 'resnext':
    resnext = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)

    def load_model(filename):
        model_dict = torch.load(filename)
        resnext.load_state_dict(model_dict)

    # load_model(MODEL_PATH+'')

    for name, param in resnext.named_parameters():
        if 'bn' in name:
            param.requires_grad = False

    out_dim = resnext.fc.weight.shape[0]

    encoder = torch.nn.Sequential(
        resnext,
        nn.ReLU(),
        nn.Linear(in_features=out_dim, out_features=key_dim)
    )
elif MODEL == 'vit':
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
    last_layer = list(vit.children())[-1].head
    last_dim = last_layer.weight.shape[0]

    print(last_dim)

    encoder = torch.nn.Sequential(
        vit,
        nn.ReLU(),
        nn.Linear(in_features=last_dim, out_features=key_dim)
    )

def save_model(model, loss, path=MODEL_PATH, identifier=''):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), 
        os.path.join(
            path,
            f"{datetime.datetime.now().__str__()[:-6].replace(':','-').replace(' ','@')[:-1]}-{identifier}-accuracy-{loss}.pth",
        ),
    )

save_model(resnet, 0.12929, MODEL_PATH, 'resnet-ct-ptinit')

# model = moco.builder.MoCo(encoder, dim=key_dim, K=dictionary_size, m=0.9999, T=0.07)

# # optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=1e-4)
# optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
#                                 lr=0.04,
#                                 momentum=0.9,
#                                 weight_decay=1e-4)

# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
#                                                             T_max=4,
#                                                             eta_min=1e-6)

# # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

# model= nn.DataParallel(model)
# model = model.to(device)

# criterion = nn.CrossEntropyLoss()

# def train_one_epoch(model, criterion, optimizer, train_dataloader):
#     model.train()
#     total_loss = 0
#     n = 0
#     total_accuracy = 0
#     for i, (q, k) in enumerate(train_dataloader):
#         q, k = q.to(device), k.to(device)
#         logits, targets = model(q, k)
#         loss = criterion(logits, targets)


#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         max_predictions = torch.argmax(logits, dim=1)
#         accuracy = (max_predictions == targets).sum().item()

#         total_accuracy += accuracy

#         total_loss += loss.detach().item()
#         n += q.size(0)

#         print(i, total_loss/n, total_accuracy/n)
    
#     lr_scheduler.step()

#     loss = total_loss/n
#     print(loss, total_accuracy/n)
#     return loss, total_accuracy/n

# for epoch in range(NUM_EPOCHS):
#     loss, accuracy = train_one_epoch(model, criterion, optimizer, train_dataloader)

#     save_model(encoder, accuracy, MODEL_PATH, MODEL + '+mlp-ct')

# if MODEL == 'resnet':
#     save_model(resnet, accuracy, MODEL_PATH, 'resnet-ct-ptinit')
# elif MODEL == 'vit':
#     save_model(vit, accuracy, MODEL_PATH, 'vit-ct-ptinit')
# elif MODEL == 'resnext':
#     save_model(resnext, accuracy, MODEL_PATH, 'resnext-ct-ptinit')
