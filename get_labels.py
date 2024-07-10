import pickle
import glob
import cv2
import numpy as np
import pandas as pd

from paths import DATA_PATH, PROCESSED_PATH

NUM_OF_VIDS = 349
FRAMES_PER_VID = 18

labels = sorted(glob.glob(DATA_PATH + '/labels/*/frame.csv'))

print(len(labels))

def get_data(labels):
    outputs = np.ones((NUM_OF_VIDS * FRAMES_PER_VID, 3), dtype=np.uint8)
    for vid_index, label in enumerate(labels):
        df = pd.read_csv(label)

        for j, row in df.iterrows():
            x1 = 1 if row.iloc[1:4].sum() > 1 else 0
            x2 = 1 if row.iloc[4:7].sum() > 1 else 0
            x3 = 1 if row.iloc[7:10].sum() > 1 else 0
            
            outputs[vid_index*18+j] = [x1, x2, x3]

    with open(PROCESSED_PATH+'/outputs', 'wb') as f:
        pickle.dump(outputs, f)

    return outputs

labels = get_data(labels)
print(len(labels))
print(labels[-1])