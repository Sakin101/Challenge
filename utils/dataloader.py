from torch.utils.data import Dataset
import os 
import torch
import cv2
import pandas as pd

class VideoLoader(Dataset):

    def __init__(self,video_path,labels_path, seq_length=150, frame_skip=150):
        self.video_path=video_path
        self.labels_path=labels_path
        self.seq_length = seq_length
        self.frame_skip = frame_skip
        self.inputs,self.output=self.get_video_data()
    
    def get_preprocessing(self,frame):
        #//TODO: add more pre_processing steps
        frame = cv2.resize(frame, dsize=(224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame=torch.tensor(frame,dtype=torch.float32)
        frame = frame/255.0
        frame = torch.permute(frame,(2,0,1))
        return frame

    def get_video_data(self):
        videos=sorted(os.listdir(self.video_path))
        labels=sorted(os.listdir(self.labels_path))
        inputs = []
        outputs = []
        for video, label in zip(videos, labels):
            video_file = os.path.join(self.video_path,video)
            label_file = os.path.join(self.labels_path, label)
            vcap = cv2.VideoCapture(video_file)
            input=[]
            for i in range(1,30*90):
                ret, frame = vcap.read()
                if not ret:
                    print("Can't receive frame")
                    break
                frame=self.get_preprocessing(frame)
                input.append(frame)
                if i % 150 == 0 and i!=0:
                    inputs.append(torch.stack(input))
                    input=[]
            df = pd.read_csv(label_file)

            for index, row in df.iterrows():
                x1 = 1 if row.iloc[1:4].sum() > 1 else 0
                x2 = 1 if row.iloc[4:7].sum() > 1 else 0
                x3 = 1 if row.iloc[7:10].sum() > 1 else 0
                outputs.append([x1, x2, x3])

            outputs = torch.tensor(outputs, dtype=torch.long)
        return inputs, outputs
    
    def __getitem__(self,i):
        return self.inputs[i],self.output[i]
    
    def __len__(self):
        return len(self.inputs)
        