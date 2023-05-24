import cv2
from sys import platform
import argparse
import sys
import os
import numpy as np
import time

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
    claheg = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    claheb = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 6))
    cllr = claher.apply(imgr)
    cllg = claheg.apply(imgg)
    cllb = claheb.apply(imgb)

    rgb_img = np.dstack((cllr, cllg, cllb))

    # 锐化图像
    kernel_edge_detection = np.array([[-1, -1, -1], [-1, 5, -1], [-1, -1, -1]])
    center_crop = cv2.filter2D(rgb_img, -1, kernel_edge_detection)
    blended = cv2.addWeighted(rgb_img, 0.5, center_crop, 0.5, 0)

    # 双边滤波
    center_crop = cv2.bilateralFilter(blended, d=5, sigmaColor=75, sigmaSpace=75)

    return center_crop


frame_count = 0  # 记录当前读取了多少帧图像
while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break
    print(frame_count)
    frame_count += 1
    if (
        frame_count != 37
        and frame_count != 104
        and frame_count != 121
        and frame_count != 149
        and frame_count != 166
        and frame_count != 166
        and frame_count != 200
        and frame_count != 239
    ):  # 每两帧图像才处理一次
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
    padding = 230
    center_crop = frame.copy()[top + padding : bottom, left:right]
    center_crop = preprocessing(center_crop.astype(np.uint8))
    resized_width, resized_height = 720, 670  # 830 800

    resized_frame = cv2.resize(center_crop, (resized_width, resized_height))
    # frame = cv2.resize(frame, (1280, 720))
    # # h, w = frame.shape[:2]

    # # # 定义输出图像的大小
    # # out_w, out_h = w, h

    # # # 定义矩形區域的大小
    # # pts_dst = np.array([[0, 0], [out_w - 2, 0], [out_w - 2, out_h - 2], [0, out_h - 1]])

    # # # 獲取 Homography 矩陣
    # # h, status = cv2.findHomography(points, pts_dst)

    # # # 進行透視變換和校正
    # # img_out = cv2.warpPerspective(frame, h, (out_w, out_h))

    # # # 顯示圖像
    # # cv2.imshow('Original Image',frame)
    # # cv2.imshow('Warped Image', img_out)
    # # cv2.waitKey(0)
    # # frame= perspective_correction(frame.copy(), points)
    # # cv2.imshow("1",frame)
    # # cv2.waitKey(0)

    # h, w = frame.shape[:2]
    # center_crop_size = min(h, w)  # 720
    # left = int((w - center_crop_size) / 2)
    # top = int((h - center_crop_size) / 2)
    # right = left + center_crop_size
    # bottom = top + center_crop_size
    # padding = 210
    # center_crop = frame.copy()[top + padding : bottom, left:right]

    # print(bottom - top, right - left)
    # resized_width, resized_height = 800,800  # 830 800

    # resized_frame = cv2.resize(center_crop , (resized_width, resized_height))

    # gray = cv2.cvtColor(resized_frame.copy(), cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 100, 300)y

    # # 矩形偵測
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area > 20000:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # # 顯示結果
    # # cv2.imshow('edges', edges)
    # cv2.imshow("result", resized_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print((bottom-top,right-left))
    # 将图像提供给 OpenPose 进行处理
    datum = op.Datum()
    datum.cvInputData = resized_frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # 获取人的坐标位置和肢体姿势信息
    people = []
    for poseKeypoints in datum.poseKeypoints:
        if len(poseKeypoints) > 0:
            if len(poseKeypoints) > 21:
                # foot_x1, foot_y1 = poseKeypoints [23][:2]
                foot_x2, foot_y2 = poseKeypoints[22][:2]
                x, y = (foot_x2), (foot_y2)
                # print('point:',x,y)
                # x,y=(foot_x1+foot_x2)/2,(foot_y1+foot_y2)/2
                player_x, player_y = x, y
            # 获取慣用腳的腳尖坐标
            foot_x, foot_y = None, None

            if len(poseKeypoints) > 21:
                foot_x, foot_y = poseKeypoints[21][:2]

            #  foot_x, foot_y = #int(foot_x * scale_x), int(foot_y * scale_y)

            # 判斷是否出現繞頭擊球
            head_x, head_y = None, None
            try:
                if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
                    head_x, head_y = poseKeypoints[0][:2]
                    # head_x, head_y = int(head_x * scale_x), int(head_y * scale_y)

                if foot_x is not None and head_x is not None and head_x < foot_x:
                    around_the_head = True
                else:
                    around_the_head = False
            except:
                around_the_head = False
            people.append((player_x, player_y, foot_x, foot_y, around_the_head))
    scale_x = (right - left) / resized_width
    scale_y = (bottom - top - padding) / resized_height
    # del frame
    datum = op.Datum()
    # 在原始图像中框出人的位置和慣用腳的腳尖
    for i, (x, y, foot_x, foot_y, around_the_head) in enumerate(people):
        # x1, y1 = max(min_w, int(x - 50)), max(min_h, int(y - 200))
        # x2, y2 = min(max_w, int(x + 50)), min(max_h, int(y + 50))
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.circle(frame, (foot_x, foot_y), 5, (0, 255, 0), -1)
        if int(y * scale_y) + top == 0:
            continue
        print(
            "Player",
            i + 1,
            "coordinates: ({}, {}), left foot: ({}, {}), around the head: {}".format(
                int(x * scale_x) + left,
                int(y * scale_y) + top + padding,
                foot_x * 2,
                foot_y * 2,
                around_the_head,
            ),
        )
        # print(x,y)
        cv2.circle(
            frame,
            (int(x * scale_x) + left, int(y * scale_y) + top + padding),
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
