import cv2
from sys import platform
import argparse
import sys
import os
import numpy as np
import time
import csv

# 設定OpenPose參數
params = dict()
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
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
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append("../../python")
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        "Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?"
    )
    raise e

# Flags
# 0-右肩, 1-右肘, 2-右腕, 3-左肩, 4-左肘,
# 5-左腕, 6-右臀, 7-右膝, 8-右踝, 9-左臀,
# 10-左膝, 11-左踝, 12-中心点, 13-颈部,
# 14-头顶, 15-鼻尖, 16-右眼, 17-左眼,
# 18-右耳, 19-左耳, 20-右脚踝, 21-右脚尖,
# 22-左脚踝, 23-左脚尖, 24-背景
# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["face"] = True
params["hand"] = True

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 讀取MP4視頻
cap = cv2.VideoCapture("C:/Users/y1595/Downloads/00001.mp4")

def perspective_correction(frame, points):
    # 取得四個點中最遠的距離
    distance = max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[1] - points[2]),
        np.linalg.norm(points[2] - points[3]),
        np.linalg.norm(points[3] - points[0]),
    )
    # 計算還原後圖片的大小
    new_width = int(distance)
    new_height = int(distance)

    # 計算還原後圖片四個點的位置
    new_points = np.array(
        [(0, 0), (new_width, 0), (new_width, new_height), (0, new_height)],
        dtype=np.float32,
    )

    # 取得透視轉換矩陣
    M = cv2.getPerspectiveTransform(points, new_points)

    # 進行透視轉換
    result = cv2.warpPerspective(frame, M, (new_width, new_height))

    return result


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


with open('TrackNetV2/prediction/00001_ball_final_predict.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    # 将 HitFrame 列表转换为整数
    hit_frames = [int(row['HitFrame']) for row in reader]

# 读取视频文件
frame_count = 0  # 记录当前读取了多少帧图像
while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count not in hit_frames:
        continue
    # 假設四個點位置，順序為左上、左下 右下、右上、
    points = np.array([(643, 577), (1291, 575), (1498, 1001), (430, 1005)])
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
    resized_width, resized_height = 720, 720

    resized_frame = cv2.resize(center_crop, (resized_width, resized_height))
    # cv2.imshow("1",resized_frame)
    # cv2.waitKey(0)
    # 将图像提供给 OpenPose 进行处理
    datum = op.Datum()
    datum.cvInputData = resized_frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # 获取人的坐标位置和肢体姿势信息
    people = []
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
                # 判斷是否出現繞頭擊球
                head = poseKeypoints[0][:2].tolist()
                maxhand = lefthand if lefthand[1] < righthand[1] else righthand
                around_the_head = True if maxhand[1] < head[1] else False
                people.append(
                    (player_x, player_y, around_the_head, lefthand, righthand, head)
                )

    scale_x = (right - left) / resized_width
    scale_y = (bottom - top - padding) / resized_height
    # del frame
    datum = op.Datum()
    # 在原始图像中框出人的位置和慣用腳的腳尖
    for i, (x, y, around_the_head, lefthand, righthand, head) in enumerate(people):
        # cv2.circle(frame, (foot_x, foot_y), 5, (0, 255, 0), -1)
        if int(y * scale_y) + top == 0:
            continue
        print(
            "Player",
            i + 1,
            f"Locations: ( {int(x * scale_x) + left}, {  int(y * scale_y) + top + padding}),  around the head: {around_the_head},headlocation=({head[0]},{head[1]})",
        )
        # print(x,y)
        cv2.circle(
            frame,
            (int(x * scale_x) + left, int(y * scale_y) + top + padding),
            5,
            (0, 255, 0),
            -1,
        ),
        cv2.circle(
            frame,
            (
                int(lefthand[0] * scale_x) + left,
                int(lefthand[1] * scale_y) + top + padding,
            ),
            5,
            (0, 255, 0),
            -1,
        ),
        cv2.circle(
            frame,
            (
                int(righthand[0] * scale_x) + left,
                int(righthand[1] * scale_y) + top + padding,
            ),
            5,
            (0, 255, 0),
            -1,
        ),
        cv2.circle(
            frame,
            (int(head[0] * scale_x) + left, int(head[1] * scale_y) + top + padding),
            5,
            (0, 255, 0),
            -1,
        )

        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", frame)
    print("Around the head shot:", any(p[4] for p in people))

    # 显示处理后的图像
    # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API",  datum.cvOutputData)
    cv2.waitKey(0)
    del frame
    datum = op.Datum()
    # time.sleep(2)
