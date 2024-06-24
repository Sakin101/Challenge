import torch, torch.nn as nn
import torchvision.models as models
from torchinfo import summary
import moco.builder

key_dim = 128

vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
last_layer = list(vit.children())[-1].head
last_dim = last_layer.weight.shape[0]

print(last_dim)

encoder = torch.nn.Sequential(
    vit,
    nn.ReLU(),
    nn.Linear(in_features=last_dim, out_features=key_dim)
)

mlp_hidden_dim=512
key_dim=128
dictionary_size=8192

model = moco.builder.MoCo(encoder, dim=key_dim, K=dictionary_size, m=0.9999, T=0.07)
summary(model, (64, 3, 224, 224))