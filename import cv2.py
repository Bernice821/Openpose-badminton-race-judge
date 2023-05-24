import cv2
import os
import pandas as pd
import numpy as np

data=r'I:\AICUP_dataset-20230507T091256Z-001\AICUP_dataset\part1\part1\train'
dest='I:\AICUP_dataset-20230507T091256Z-001\AICUP_dataset\All'
for i in os.listdir(data):

    foldername=os.path.join(data,i)
    folder=os.listdir(foldername)
    
    csv=''
    mp4=''
    for f in folder:
        if '.csv' in f:
            csv=f
        if '.mp4' in f:
            mp4=f
    
    data=pd.read_csv(os.path.join(foldername,csv))
  
    framei=0
    print(os.path.join(foldername,mp4))
    cap = cv2.VideoCapture(os.path.join(foldername,mp4))
    
    while cap.isOpened():
    # 读取一帧
        ret, frame = cap.read()
        framei+=1
        if(framei in data['ShotSeq']):
            frame = cv2.resize(frame, (1280, 720)) 
            cv2.imwrite(frame,os.listdir(dest,(i+data['HitFrame']+".jpg")))
    break

    
