import pickle
import torch
import numpy as np

with open('./videos/inputs', 'rb') as f:
    frames = pickle.load(f)

frames = frames[:349*90*5]

with open('./videos/inputs', 'wb') as f:
    pickle.dump(frames, f)
