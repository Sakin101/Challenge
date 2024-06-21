import os
import torch
from torch.utils.data import DataLoader
from utils.dataloader import VideoLoader
from models.model import VideoClassificationModel
device="cuda" if torch.cuda.is_available() else "cpu"
print(device)
data_set=VideoLoader("video","labels")
train_data=DataLoader(dataset=data_set,batch_size=1,shuffle=True)
model=VideoClassificationModel(256,2,1,1,0.5,3).to(device)
for x,y in train_data:
    print(x.shape)
    print(y.shape)
    model.eval()
    with torch.no_grad():
        output=model(x.to(device),None)
    print(output)
    break

print(data_set)
