import math
import csv
import numpy as np
import cv2


def is_point_inside_quadrilateral(quadrilateral_points, new_point):
    # 轉換座標點為NumPy陣列
    points = np.array(quadrilateral_points, dtype=np.float32)

    # 使用cv2.pointPolygonTest函數判斷新的點是否在四邊形內部
    result = cv2.pointPolygonTest(points, new_point, False)

    if result >= 0:
        return True
    else:
        return False

def caculatewinner(court, ball_point):
    # for i , court in enumerate(court_point):
    quadrilateral_points = [
        (int(court.loc[0, "up_left"][0]), int(court.loc[0, "up_left"][1])),
        (int(court.loc[0, "up_right"][0]), int(court.loc[0, "up_right"][1])),
        (int(court.loc[0, "down_left"][0]), int(court.loc[0, "down_left"][1])),
        (int(court.loc[0, "down_right"][0]), int(court.loc[0, "down_right"][1])),
    ]
    print("Quadrilateral Points:", quadrilateral_points)
    print("Image Size:", ball_point)

    if ball_point[1] <= int(court.loc[0, "mid_left"][1]):
        # 落點在上半場
        print("落點在上半場")
        if is_point_inside_quadrilateral(quadrilateral_points, ball_point):
            print("A win")
            return "A"
        else:
            return "B"
            print("B win")
    else:
        # 落點在下半場
        print("落點在下半場")
        if is_point_inside_quadrilateral(quadrilateral_points, ball_point):
            print("B win")
            return "B"
        else:
            print("A win")
            return "A"
