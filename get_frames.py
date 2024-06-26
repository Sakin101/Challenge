import pickle
import glob
import cv2
import numpy as np
import pandas as pd

from paths import DATA_PATH, PROCESSED_PATH

NUM_OF_VIDS = 200
FRAMES_PER_VID = 18

videos = sorted(glob.glob(DATA_PATH + '/videos/*'))[0:NUM_OF_VIDS]
labels = sorted(glob.glob(DATA_PATH + '/labels/*/frame.csv'))[0:NUM_OF_VIDS]

def get_data(videos, labels):
    inputs = np.zeros((NUM_OF_VIDS * FRAMES_PER_VID, 3, 224, 224), dtype=np.uint8)
    outputs = np.zeros((NUM_OF_VIDS * FRAMES_PER_VID, 3), dtype=np.uint8)
    x = 0
    for vid_index, (video, label) in enumerate(zip(videos, labels)):
        vcap = cv2.VideoCapture(video)
        for i in range(0, 30*90, 150):
            vcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vcap.read()
            # if i % 150 != 0:
            #     continue
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv2.resize(frame, dsize=(224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))
            inputs[x] = frame.astype(np.uint8)
            x+=1

        df = pd.read_csv(label)

        for j, row in df.iterrows():
            x1 = 1 if row.iloc[1:4].sum() > 1 else 0
            x2 = 1 if row.iloc[4:7].sum() > 1 else 0
            x3 = 1 if row.iloc[7:10].sum() > 1 else 0
            
            outputs[vid_index*18+j] = [x1, x2, x3]

    with open(PROCESSED_PATH+'/inputs', 'wb') as f:
        pickle.dump(inputs, f)

    # with open(PROCESSED_PATH+'/outputs', 'wb') as f:
    #     pickle.dump(outputs, f)

    return inputs, outputs

inputs, labels = get_data(videos, labels)
