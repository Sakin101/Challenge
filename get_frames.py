import pickle
import torch
import glob
import cv2
import pandas as pd

from paths import DATA_PATH, PROCESSED_PATH

videos = glob.glob(DATA_PATH + '/videos/*')
labels = glob.glob(DATA_PATH + '/labels/*/frame.csv')


def get_data(videos, labels):
    inputs = torch.zeros(0, 3, 224, 224)
    outputs = []
    for video, label in zip(videos, labels):
        print(video)
        vcap = cv2.VideoCapture(video)
        for i in range(30*90):
            ret, frame = vcap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv2.resize(frame, dsize=(224, 224))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.tensor(frame, dtype=torch.float32)
            t = t/255*2-1
            t = torch.permute(t, (2, 0, 1))
            if i % 150 == 0:
                inputs = torch.cat((inputs, t.unsqueeze(0)), dim=0)

        df = pd.read_csv(label)

        for index, row in df.iterrows():
            x1 = 1 if row.iloc[1:4].sum() > 1 else 0
            x2 = 1 if row.iloc[4:7].sum() > 1 else 0
            x3 = 1 if row.iloc[7:10].sum() > 1 else 0
            outputs.append([x1, x2, x3])

    outputs = torch.tensor(outputs, dtype=torch.long)

    with open(PROCESSED_PATH+'/inputs', 'wb') as f:
        pickle.dump(inputs, f)

    with open(PROCESSED_PATH+'/outputs', 'wb') as f:
        pickle.dump(outputs, f)

    return inputs, outputs

get_data(videos, labels)