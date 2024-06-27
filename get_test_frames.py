import pickle
import glob
import cv2
import numpy as np
import pandas as pd

from paths import TEST_DATA_PATH, TEST_PROCESSED_PATH

NUM_OF_VIDS = 100
FRAMES_PER_VID = 18

videos = sorted(glob.glob(TEST_DATA_PATH + '/videos/*'))[0:NUM_OF_VIDS]

def get_data(videos):
    inputs = np.zeros((NUM_OF_VIDS * FRAMES_PER_VID, 3, 224, 224), dtype=np.uint8)
    x = 0
    for vid_index, video in enumerate(videos):
        print(video)
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

    with open(TEST_PROCESSED_PATH+'/inputs', 'wb') as f:
        pickle.dump(inputs, f)

    return inputs

inputs = get_data(videos)
