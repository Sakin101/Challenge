import torch 
import torch.nn as nn
from torchvision import models
class FeatureExtraction(nn.Module):
    def __init__(self,embed_size):
        super(FeatureExtraction,self).__init__()

        resnet=models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children()))

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(resnet.fc.in_features,embed_size)
    
    def forward(self,x):
        batch_size, seq_length,channel,height,width=x.shape
        x=x.view(batch_size*seq_length,3,height,width)
        x=self.feature_extractor(x)
        x=self.avg_pool(x)
        x=torch.flatten(x,1)
        x=self.fc(x)
        x=x.view(batch_size,seq_length,-1)

class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads):
        super(SelfAttention,self).__init__()
        self.embed_size=embed_size
        self.heads=heads
        self.head_dim= embed_size // heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self,values,keys,query,mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerDecoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class VideoClassificationModel(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, num_classes):
        super(VideoClassificationModel, self).__init__()
        self.cnn = FeatureExtraction(embed_size)
        self.decoder = TransformerDecoder(embed_size, num_layers, heads, forward_expansion, dropout)
        self.classification_head = nn.Linear(embed_size, num_classes)
        
    def forward(self, x, mask):
        x = self.cnn(x)  
        x = self.decoder(x, mask)  
        x = x.mean(dim=1)  
        return self.classification_head(x)