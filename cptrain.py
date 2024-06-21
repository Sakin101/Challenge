import torch
import pickle

import torchvision.transforms.v2 as v2

import moco.loader
import moco.builder

import torch.nn as nn

with open('./videos/inputs', 'rb') as f:
    inputs = (pickle.load(f))[:1024]

BATCH_SIZE=128
NUM_EPOCHS=5

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.inputs = torch.tensor(inputs, dtype=torch.uint8)
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.transform(self.inputs[i])

normalize = v2.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
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

train_dataset = Dataset(transform=moco.loader.TwoCropsTransform(v2.Compose(augmentation)))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

import torchvision.models as models

mlp_hidden_dim=1024
key_dim=128
dictionary_size=65536
device='cuda'

resnet = models.resnet50()
last_layer = list(resnet.children())[-1]
avg_pool_dim = last_layer.weight.shape[1]
print(avg_pool_dim)

layers = list(resnet.children())[:-1]
layers.append(torch.nn.Flatten())

layers.extend(
            [
            nn.Linear(in_features=avg_pool_dim, out_features=mlp_hidden_dim), 
            nn.ReLU(), 
            nn.Sequential(nn.Linear(in_features=mlp_hidden_dim, out_features=key_dim))
            ]
        )

encoder = torch.nn.Sequential(*layers)

model = moco.builder.MoCo(encoder, dim=key_dim, K=dictionary_size, m=0.9999, T=0.07)
a = torch.randn((3, 3, 224, 224))
b = model.encoder_q(a)
print(b.grad_fn)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, criterion, optimizer, train_dataloader):
    model.train()
    for q, k in train_dataloader:
        q, k = q.to(device), k.to(device)
        logits, targets = model(q, k)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

for epoch in range(NUM_EPOCHS):
    train_one_epoch(model, criterion, optimizer, train_dataloader)
