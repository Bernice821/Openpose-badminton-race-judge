
import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Badminton_Dataset
from utils import *
from denoise import smooth
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.signal import find_peaks
import argparse
from sys import platform
import argparse
import sys
import numpy as np
import court_superglue
import winner_detection
# 設定OpenPose參數
params = dict()
dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
folder='I:\\AICUP_dataset-20230507T091256Z-001\\AICUP_dataset\\part1\\part1\\val\\'

for i in os.listdir(folder):
    tmppath=os.path.join(folder,i)
    for j in os.listdir(tmppath):
        print(os.path.join(tmppath,j))
  
parser.add_argument("--video_file", type=str)
parser.add_argument("--model_file", type=str, default="TrackNetV2/models/model_best.pt")
parser.add_argument("--num_frame", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

video_file = args.video_file
model_file = args.model_file
num_frame = args.num_frame
batch_size = args.batch_size

video_name = (video_file.replace('\\','/').split("/"))[-1]#[:-4]
video_format = video_file.split("/")[-1][-3:]
# print(video_name)
if video_format == "avi":
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
elif video_format == "mp4":
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
else:
    raise ValueError("Invalid video format.")

checkpoint = torch.load(model_file)
param_dict = checkpoint["param_dict"]
model_name = param_dict["model_name"]
num_frame = param_dict["num_frame"]
input_type = param_dict["input_type"]

# Load model
model = get_model(model_name, num_frame, input_type).cuda()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

cap = cv2.VideoCapture(args.video_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 

success = True
frame_count = 0
num_final_frame = 0
ratio = h / HEIGHT
columns = ["Frame", "Visibility", "X", "Y"]
df = pd.DataFrame(columns=columns)
while success:
    # Sample frames to form input sequence
    frame_queue = []
    for _ in range(num_frame * batch_size):
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_count += 1
            frame_queue.append(frame)
    if not frame_queue:
        break
    if len(frame_queue) % num_frame != 0:
        frame_queue = []
        # Record the length of remain frames
        num_final_frame = len(frame_queue)
        # Adjust the sample timestampe of cap
        frame_count = frame_count - num_frame * batch_size
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        # Re-sample mini batch
        for _ in range(num_frame * batch_size):
            success, frame = cap.read()
            if not success:
                break
            else:
                frame_count += 1
                frame_queue.append(frame)
        assert len(frame_queue) % num_frame == 0

    x = get_frame_unit(frame_queue, num_frame)
    # Inference
    with torch.no_grad():
        y_pred = model(x.cuda())
    y_pred = y_pred.detach().cpu().numpy()
    h_pred = y_pred > 0.5
    h_pred = h_pred * 255.0
    h_pred = h_pred.astype("uint8")
    h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

    for i in range(h_pred.shape[0]):
        if num_final_frame > 0 and i < (num_frame * batch_size - num_final_frame):
            continue
        else:
            img = frame_queue[i].copy()
            cx_pred, cy_pred = get_object_center(h_pred[i])
            cx_pred, cy_pred = int(ratio * cx_pred), int(ratio * cy_pred)
            vis = 1 if cx_pred > 0 and cy_pred > 0 else 0

            df.loc[len(df)] = [
                frame_count - (num_frame * batch_size) + i,
                vis,
                cx_pred,
                cy_pred,
            ]

df = smooth(df)
def event_detection(df):
    frames = []
    realx = []
    realy = []
    points = []

    def angle(v1, v2):
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / math.pi)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle


    def get_point_line_distance(point, line):
        point_x = point[0]
        point_y = point[1]
        line_s_x = line[0]
        line_s_y = line[1]
        line_e_x = line[2]
        line_e_y = line[3]
        if line_e_x - line_s_x == 0:
            return math.fabs(point_x - line_s_x)
        if line_e_y - line_s_y == 0:
            return math.fabs(point_y - line_s_y)
        # 斜率
        k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
        # 截距
        b = line_s_y - k * line_s_x
        # 带入公式得到距离dis
        dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
        return dis


    num = 0
    count = 0
    front_zeros = np.zeros(len(df))
    for i in range(0, len(df)):
        frames.append(int(float(df.iloc[i, 0])))
        realx.append(int(float(df.iloc[i, 2])))
        realy.append(int(float(df.iloc[i, 3])))
        if int(float(df.iloc[i, 2])) != 0:
            front_zeros[num] = count
            points.append(
                (
                    int(float(df.iloc[i, 2])),
                    int(float(df.iloc[i, 3])),
                    int(float(df.iloc[i, 0])),
                )
            )
            num += 1
        else:
            count += 1

    # 羽球2D軌跡點
    points = np.array(points)
    x, y, z = points.T

    Predict_hit_points = np.zeros(len(frames))
    ang = np.zeros(len(frames))
    # from scipy.signal import find_peaks
    peaks, properties = find_peaks(y, prominence=5)
    if len(peaks) >= 5:
        lower = np.argmin(y[peaks[0] : peaks[1]])
        if (y[peaks[0]] - lower) < 5:
            peaks = np.delete(peaks, 0)

        lower = np.argmin(y[peaks[-2] : peaks[-1]])
        if (y[peaks[-1]] - lower) < 5:
            peaks = np.delete(peaks, -1)
    
    start_point = 0

    for i in range(len(y) - 1):
        if (y[i] - y[i + 1]) / (z[i + 1] - z[i]) >= 5:
            start_point = i + front_zeros[i]
            Predict_hit_points[int(start_point)] = 1
            print(int(start_point))
            break

    end_point = 10000
    for i in range(len(peaks)):
        print(peaks[i] + int(front_zeros[peaks[i]]), end=",")
        if (
            peaks[i] + int(front_zeros[peaks[i]]) >= start_point
            and peaks[i] + int(front_zeros[peaks[i]]) <= end_point
        ):
            Predict_hit_points[peaks[i] + int(front_zeros[peaks[i]])] = 1

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1] + 1
        upper = []
        # plt.plot(z[start:end],y[start:end]*-1,'-')
        lower = np.argmin(y[start:end])  # 找到最低谷(也就是從最高點開始下墜到下一個擊球點),以此判斷扣殺或平球軌跡
        for j in range(start + lower, end + 1):
            if (j - (start + lower) > 5) and (end - j > 5):
                if (y[j] - y[j - 1]) * 3 < (y[j + 1] - y[j]):
                    print(j, end=",")
                    ang[j + int(front_zeros[j])] = 1

                point = [x[j], y[j]]
                line = [x[j - 1], y[j - 1], x[j + 1], y[j + 1]]
                # if get_point_line_distance(point,line) > 2.5:
                if (
                    angle(
                        [x[j - 1], y[j - 1], x[j], y[j]], [x[j], y[j], x[j + 1], y[j + 1]]
                    )
                    > 130
                ):
                    print(j, end=",")
                    ang[j + int(front_zeros[j])] = 1

    ang, _ = find_peaks(ang, distance=15)
    # final_predict, _  = find_peaks(Predict_hit_points, distance=10)
    for i in ang:
        Predict_hit_points[i] = 1
    Predict_hit_points, _ = find_peaks(Predict_hit_points, distance=5)
    final_predict = []
    for i in Predict_hit_points:
        final_predict.append(i)

    with open("final_predict.csv", "w", newline="") as csvfile1:
        h = csv.writer(csvfile1)
        h.writerow(["ShotSeq", "HitFrame"])
        for i, item in enumerate(final_predict):
            h.writerow([i + 1, item])
    return final_predict

final_predict=event_detection(df)
Xpos = df.loc[final_predict, 'X']
Ypos=df.loc[final_predict,'Y']
 
def preprocessing(img):
    imgr = img[:, :, 0]
    imgg = img[:, :, 1]
    imgb = img[:, :, 2]

    claher = cv2.createCLAHE(clipLimit=3, tileGridSize=(6, 6))
    claheg = cv2.createCLAHE(clipLimit=3, tileGridSize=(6, 6))
    claheb = cv2.createCLAHE(clipLimit=3, tileGridSize=(6, 6))
    cllr = claher.apply(imgr)
    cllg = claheg.apply(imgg)
    cllb = claheb.apply(imgb)

    img = np.dstack((cllr, cllg, cllb))

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.medianBlur(img, 5)

    # 銳化圖像
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    center_crop = cv2.filter2D(img, -1, kernel)

    return center_crop
 
del model  # 刪除模型
torch.cuda.empty_cache()  # 釋放 GPU 內存
del df
del cap
del checkpoint
del parser 
del params
del param_dict
frame_count = 0  # 记录当前读取了多少帧图像
cap = cv2.VideoCapture(video_file)
try:
    # Windows Import
    if platform == "win32":
        sys.path.append(dir_path + "/../../python/openpose/Release")
        os.environ["PATH"] = (
            os.environ["PATH"]
            + ";"
            + dir_path
            + "/../../x64/Release;"
            + dir_path
            + "/../../bin;"
        )
        import pyopenpose as op
    else:
        sys.path.append("../../python")
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        "Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?"
    )
    raise e
params = dict()
params["model_folder"] = "../../../models/"
params["face"] = True
params["hand"] = True
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()



columns=["VideoName","ShotSeq","HitFrame","Hitter",
"RoundHead","Backhand","BallHeight","LandingX","LandingY",
"HitterLocationX","HitterLocationY","DefenderLocationX","DefenderLocationY","BallType","Winner"]
finaldf = pd.DataFrame(columns=columns)
count=1
while cap.isOpened():
    # 读取一帧      
  
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count in final_predict:  # 每两帧图像才处理一次
        
        shotseq=final_predict.index(frame_count)+1
        print(Xpos)
        ball_x = Xpos[ frame_count ]  # 假設您已經有球的x座標資訊
        ball_y = Ypos[ frame_count ]  # 假設您已經有球的y座標資訊



        min_distance = float('inf')
        hitter = None
        defender = None
        _distence=[]
        frame = cv2.resize(frame, (1280, 720))
        h, w = frame.shape[:2]
        center_crop_size = min(h, w)  # 720
        left = int((w - center_crop_size) / 2)
        top = int((h - center_crop_size) / 2)
        right = left + center_crop_size
        bottom = top + center_crop_size
        padding = 235
        center_crop = frame.copy()[top + padding : bottom, left:right]
        center_crop = preprocessing(center_crop.astype(np.uint8))
        resized_width, resized_height =520,520

        resized_frame = cv2.resize(center_crop, (resized_width, resized_height))

        datum = op.Datum()
        datum.cvInputData = resized_frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # 获取人的坐标位置和肢体姿势信息
        people = []
        # 設定中心點座標
        center_y = (resized_frame.shape[0] // 2)

        for poseKeypoints in datum.poseKeypoints:
            if len(poseKeypoints) > 0:
            # 获取慣用腳的腳尖坐标
                if len(poseKeypoints) > 21:
                    foot_x2, foot_y2 = poseKeypoints[22][:2]
                    x, y = (foot_x2), (foot_y2)
                    player_x, player_y = x, y
                    lefthand, righthand = (
                        poseKeypoints[7][:2].tolist(),
                        poseKeypoints[4][:2].tolist(),
                    )
                    leftshoulder = poseKeypoints[5][:2].tolist()
                    rightshoulder = poseKeypoints[2][:2].tolist()

                    # 判斷是否出現繞頭擊球
                    head = poseKeypoints[0][:2].tolist()
                    maxhand = lefthand if lefthand[1] < righthand[1] else righthand
                    around_the_head = True if maxhand[1] < head[1] else False

                    # 判斷人物座標位置，標記為A或B
                    if player_y < center_y:
                        is_forehand = dominant_hand[0] < non_dominant_hand[0]
                        people.append(
                            (player_x, player_y, around_the_head, lefthand, righthand, head, "A",leftshoulder,rightshoulder)
                        )
                    else:
                        dominant_hand = lefthand if lefthand[1] > righthand[1] else righthand
                        non_dominant_hand = lefthand if lefthand[1] <= righthand[1] else righthand
                        is_forehand = dominant_hand[0] > non_dominant_hand[0]
                        people.append(
                            (player_x, player_y, around_the_head, lefthand, righthand, head, "B",leftshoulder,rightshoulder)
                        )
                    
                    _distence.append(distance(player_x,player_y,ball_x,ball_y))
        if len(people)<2:
            continue
        hitter=people[0][6] if _distence[0]<_distence[1] else people[1][6]
        defender=people[0][6] if _distence[0]>_distence[1] else people[1][6]
        print("Hitter:",hitter)
        for i in range(len(people)):
            if people[i][6]==hitter:
                lefthand = people[i][3]
                righthand = people[i][4]
                leftshoulder = people[i][7]
                rightshoulder = people[i][8]
                locationX=people[i][0]
                locationY=people[i][1]

                is_forehand = (lefthand[0] > leftshoulder[0] and righthand[0] > rightshoulder[0]) or (lefthand[0] < leftshoulder[0] and righthand[0] < rightshoulder[0])
                backhand=1 if is_forehand else 2
                print("正反手:", "正手" if is_forehand else "反手")
            else:
                defendX=people[i][0]
                defendY=people[i][1]
        scale_x = (right - left) / resized_width
        scale_y = (bottom - top - padding) / resized_height
        import random
        if frame_count==final_predict[-1]:
            courtdf=court_superglue.court_superglue(resized_frame)
            print(courtdf)
            winner=winner_detection.caculatewinner(courtdf,(int(ball_x),int(ball_y)))
            finaldf=finaldf.append(pd.Series([video_name,count,frame_count,hitter,2,backhand,random.randint(1,2),ball_x,ball_y,locationX,locationY,defendX,defendY,random.randint(1, 9),winner],index=finaldf.columns), ignore_index=True)
        else:
            finaldf=finaldf.append(pd.Series([video_name,count,frame_count,hitter,2,backhand,random.randint(1,2),ball_x,ball_y,locationX,locationY,defendX,defendY,random.randint(1, 9),'X'],index=finaldf.columns), ignore_index=True)
        count+=1

        datum = op.Datum()
        # 在原始图像中框出人的位置和慣用腳的腳尖
        for i, (x, y, around_the_head, lefthand, righthand, head,person,leftshoulder,rightshoulder)  in enumerate(people):
            # cv2.circle(frame, (foot_x, foot_y), 5, (0, 255, 0), -1)
            if int(y * scale_y) + top == 0:
                continue
            print(
                f"Player{person}:",
                f"Locations: ( {int(x * scale_x) + left}, {  int(y * scale_y) + top + padding}),  around the head: {around_the_head},headlocation=({head[0]},{head[1]}) defender={defender} hitter={hitter}",
            )
        
        max_mem_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # in MB
        print(f"Maximum GPU memory used: {max_mem_used} MB")
        
      
        cv2.waitKey(0)
        del frame
        del datum
        torch.cuda.empty_cache()  # 釋放 GPU 內存
       
print(finaldf)
finaldf.to_csv(f"result.csv",index=False)
