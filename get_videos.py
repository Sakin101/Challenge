import pickle
import glob
import cv2
import numpy as np

from paths import DATA_PATH, VIDEO_PATH2

NUM_OF_VIDS = 200
FRAMES_PER_SEC = 5
FRAMES_PER_VID = 90 * FRAMES_PER_SEC

videos = sorted(glob.glob(DATA_PATH + '/videos/*'))[0:NUM_OF_VIDS]

def get_data(videos):
    inputs = np.zeros((len(videos) * FRAMES_PER_VID, 3, 224, 224), dtype=np.uint8)
    x = 0
    for video in videos:
        print(video)
        vcap = cv2.VideoCapture(video)
        for i in range(30*90):
            # vcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vcap.read()
            if i % (30//FRAMES_PER_SEC) != 0:
                continue
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv2.resize(frame, dsize=(224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))
            inputs[x] = frame.astype(np.uint8)
            x+=1

    print(inputs.shape, inputs.dtype)
    with open(VIDEO_PATH2+'/inputs', 'wb') as f:
        pickle.dump(inputs, f)

get_data(videos)