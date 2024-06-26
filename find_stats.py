import cv2
import glob
import numpy as np
from paths import DATA_PATH

NUM_OF_VIDS = 200

vids = glob.glob(DATA_PATH+'/videos')

reds = np.zeros((18*224*224*NUM_OF_VIDS))
greens = np.zeros((18*224*224*NUM_OF_VIDS))
blues = np.zeros((18*224*224*NUM_OF_VIDS))

videos = sorted(glob.glob(DATA_PATH + '/videos/*'))[:NUM_OF_VIDS]
def get_data(videos):
    x = 0
    for video in videos:
        print(video)
        vcap = cv2.VideoCapture(video)
        for i in range(30*90):
            # vcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vcap.read()
            if i % 150 != 0:
                continue
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv2.resize(frame, dsize=(224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))
            frame = frame/255

            reds[x:x+224*224] = frame[0].flatten()
            greens[x:x+224*224] = frame[1].flatten()
            blues[x:x+224*224] = frame[2].flatten()

            x+=224*224

    return x

x = get_data(videos)
print(x, len(reds))
print(np.mean(reds), np.mean(greens), np.mean(blues))
print(np.std(reds), np.std(greens), np.std(blues))