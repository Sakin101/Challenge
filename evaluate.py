import torch
from sklearn.metrics import average_precision_score
import numpy as np
from tqdm import tqdm
import torch.nn as nn

def evaluate_map(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (images, labels) in (enumerate(dataloader)):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs)  # Sigmoid to get probabilities
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate average precision for each label
    ap_per_label = []
    for i in range(all_labels.shape[1]):
        ap = average_precision_score(all_labels[:, i], all_preds[:, i])
        ap_per_label.append(ap)
    map_score = np.mean(ap_per_label)
    return map_score, ap_per_label