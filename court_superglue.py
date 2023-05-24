from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import csv
import torch
import numpy as np
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import (
    AverageTimer,
    VideoStreamer,
    make_matching_plot_fast,
    frame2tensor,
)
import pandas as pd
from torchvision.transforms import functional as F
from PIL import Image


def image2tensor(image, device):
    # 将图像从 BGR 转换为 RGB 格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将图像转换为浮点类型并归一化到 0-1 范围
    image_normalized = image_rgb.astype(np.float32) / 255.0
    # 将图像的通道顺序从 H x W x C 转换为 C x H x W
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    # 创建一个只有一个通道的图像张量
    image_tensor = torch.from_numpy(image_transposed[0:1]).unsqueeze(0)
    # 将图像张量移动到指定设备
    tensor = image_tensor.to(device)
    return tensor


 
torch.set_grad_enabled(False)


def court_superglue(image):
    columns = [
        "up_left",
        "mid_left",
        "down_left",
        "up_right",
        "mid_right",
        "down_right",
    ]
    df = pd.DataFrame(columns=columns)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Running inference on device "{}"'.format(device))
    config = {
        "superpoint": {
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": -1,
        },
        "superglue": {
            "weights": "indoor",
            "sinkhorn_iterations": 20,
            "match_threshold": 0.2,
        },
    }
    matching = Matching(config).eval().to(device)
    keys = ["keypoints", "scores", "descriptors"]

    # vs = VideoStreamer(videopath, [640,480], 1,
    #                   ['*.png', '*.jpg', '*.jpeg'], 1000000)
    # frame, ret = vs.next_frame()
    # assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = image2tensor(image, device)
    last_data = matching.superpoint({"image": frame_tensor})
    last_data = {k + "0": last_data[k] for k in keys}
    last_data["image0"] = frame_tensor
    last_frame = image  # frame
    # last_image_id = 0
    T_indx = []

    # while True:
        # frame, ret = vs.next_frame()
        # if not ret:
        #     print('Finished court_superglue.py')
        #     break
        
    frame_tensor = image2tensor(image, device)
    # stem0, stem1 = last_image_id, vs.i - 1
    # frame_tensor = frame2tensor(frame, device)
    pred = matching({**last_data, "image1": frame_tensor})
    kpts0 = last_data["keypoints0"][0].cpu().numpy()
    kpts1 = pred["keypoints1"][0].cpu().numpy()
    matches = pred["matches0"][0].cpu().numpy()
    confidence = pred["matching_scores0"][0].cpu().numpy()
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    # color = cm.jet(confidence[valid])
    # text = [
    #     'SuperGlue',
    #     'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
    #     'Matches: {}'.format(len(mkpts0))
    # ]
    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
    # 使用 cv2.perspectiveTransform() 函数变换点集
    points = np.float32(
        [[428, 385], [385, 485], [289, 668], [861, 385], [910, 485], [995, 668]]
    ).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, H)
    # 轉換為 Python list 並取出 [x, y] 座標點
    transformed_points = [
        [point[0][0], point[0][1]] for point in transformed_points.tolist()
    ]
    # print(transformed_points)
    T_indx.append(transformed_points)
    # print("-------------")
    # k_thresh = matching.superpoint.config['keypoint_threshold']
    # m_thresh = matching.superglue.config['match_threshold']
    # small_text = [
    #     'Keypoint Threshold: {:.4f}'.format(k_thresh),
    #     'Match Threshold: {:.2f}'.format(m_thresh),
    #     'Image Pair: {:06}:{:06}'.format(stem0, stem1),
    # ]

    for i, x in enumerate(T_indx):
        df = df.append(
            pd.Series([x[0], x[1], x[2], x[3], x[4], x[5]], index=df.columns),
            ignore_index=True,
        )

    df.to_excel("c1.xlsx")
    torch.cuda.empty_cache()  # 釋放 GPU 內存
    # vs.cleanup()
    # print(df)
    return df


# torch.set_grad_enabled(False)
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='SuperGlue demo',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument(
#         '--input', type=str, default='0',
#         help='ID of a USB webcam, URL of an IP camera, '
#              'or path to an image directory or movie file')
#     parser.add_argument(
#         '--output_dir', type=str, default=None,
#         help='Directory where to write output frames (If None, no output)')

#     parser.add_argument(
#         '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
#         help='Glob if a directory of images is specified')
#     parser.add_argument(
#         '--skip', type=int, default=1,
#         help='Images to skip if input is a movie or directory')
#     parser.add_argument(
#         '--max_length', type=int, default=1000000,
#         help='Maximum length if input is a movie or directory')
#     parser.add_argument(
#         '--resize', type=int, nargs='+', default=[640, 480],
#         help='Resize the input image before running inference. If two numbers, '
#              'resize to the exact dimensions, if one number, resize the max '
#              'dimension, if -1, do not resize')

#     parser.add_argument(
#         '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
#         help='SuperGlue weights')
#     parser.add_argument(
#         '--max_keypoints', type=int, default=-1,
#         help='Maximum number of keypoints detected by Superpoint'
#              ' (\'-1\' keeps all keypoints)')
#     parser.add_argument(
#         '--keypoint_threshold', type=float, default=0.005,
#         help='SuperPoint keypoint detector confidence threshold')
#     parser.add_argument(
#         '--nms_radius', type=int, default=4,
#         help='SuperPoint Non Maximum Suppression (NMS) radius'
#         ' (Must be positive)')
#     parser.add_argument(
#         '--sinkhorn_iterations', type=int, default=20,
#         help='Number of Sinkhorn iterations performed by SuperGlue')
#     parser.add_argument(
#         '--match_threshold', type=float, default=0.2,
#         help='SuperGlue match threshold')

#     parser.add_argument(
#         '--show_keypoints', action='store_true',
#         help='Show the detected keypoints')
#     parser.add_argument(
#         '--no_display', action='store_true',
#         help='Do not display images to screen. Useful if running remotely')
#     parser.add_argument(
#         '--force_cpu', action='store_true',
#         help='Force pytorch to run in CPU mode.')

#     opt = parser.parse_args()
#     print(opt)

#     if len(opt.resize) == 2 and opt.resize[1] == -1:
#         opt.resize = opt.resize[0:1]
#     if len(opt.resize) == 2:
#         print('Will resize to {}x{} (WxH)'.format(
#             opt.resize[0], opt.resize[1]))
#     elif len(opt.resize) == 1 and opt.resize[0] > 0:
#         print('Will resize max dimension to {}'.format(opt.resize[0]))
#     elif len(opt.resize) == 1:
#         print('Will not resize images')
#     else:
#         raise ValueError('Cannot specify more than two integers for --resize')

#     device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
#     print('Running inference on device \"{}\"'.format(device))
#     config = {
#         'superpoint': {
#             'nms_radius': opt.nms_radius,
#             'keypoint_threshold': opt.keypoint_threshold,
#             'max_keypoints': opt.max_keypoints
#         },
#         'superglue': {
#             'weights': opt.superglue,
#             'sinkhorn_iterations': opt.sinkhorn_iterations,
#             'match_threshold': opt.match_threshold,
#         }
#     }
# matching = Matching(config).eval().to(device)
# keys = ['keypoints', 'scores', 'descriptors']

# vs = VideoStreamer(opt.input, opt.resize, opt.skip,
#                    opt.image_glob, opt.max_length)
# frame, ret = vs.next_frame()
# assert ret, 'Error when reading the first frame (try different --input?)'

# frame_tensor = frame2tensor(frame, device)
# last_data = matching.superpoint({'image': frame_tensor})
# last_data = {k+'0': last_data[k] for k in keys}
# last_data['image0'] = frame_tensor
# last_frame = frame
# last_image_id = 0

# if opt.output_dir is not None:
#     print('==> Will write outputs to {}'.format(opt.output_dir))
#     Path(opt.output_dir).mkdir(exist_ok=True)

# timer = AverageTimer()
# T_indx = []

# while True:
#     frame, ret = vs.next_frame()
#     if not ret:
#         print('Finished court_superglue.py')
#         break
#     timer.update('data')
#     stem0, stem1 = last_image_id, vs.i - 1

#     frame_tensor = frame2tensor(frame, device)
#     pred = matching({**last_data, 'image1': frame_tensor})
#     kpts0 = last_data['keypoints0'][0].cpu().numpy()
#     kpts1 = pred['keypoints1'][0].cpu().numpy()
#     matches = pred['matches0'][0].cpu().numpy()
#     confidence = pred['matching_scores0'][0].cpu().numpy()
#     timer.update('forward')

#     valid = matches > -1
#     mkpts0 = kpts0[valid]
#     mkpts1 = kpts1[matches[valid]]
#     color = cm.jet(confidence[valid])
#     text = [
#         'SuperGlue',
#         'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
#         'Matches: {}'.format(len(mkpts0))
#     ]
#     # print(text[1])
#     print("--------------------")
#     # print(mkpts0)
#     print("--------------------")

#     H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
#     # 使用 cv2.perspectiveTransform() 函数变换点集
#     points = np.float32([[428, 385], [385, 485], [289, 668], [861, 385], [910,485], [995,668]]).reshape(-1, 1, 2)
#     transformed_points = cv2.perspectiveTransform(points, H)
#     # 轉換為 Python list 並取出 [x, y] 座標點
#     transformed_points = [[point[0][0], point[0][1]] for point in transformed_points.tolist()]
#     print(transformed_points)
#     T_indx.append(transformed_points)
#     print("-------------")
#     k_thresh = matching.superpoint.config['keypoint_threshold']
#     m_thresh = matching.superglue.config['match_threshold']
#     small_text = [
#         'Keypoint Threshold: {:.4f}'.format(k_thresh),
#         'Match Threshold: {:.2f}'.format(m_thresh),
#         'Image Pair: {:06}:{:06}'.format(stem0, stem1),
#     ]
#     out = make_matching_plot_fast(
#         last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
#         path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

#     if not opt.no_display:
#         # cv2.imshow('SuperGlue matches', out)
#         key = chr(cv2.waitKey(1) & 0xFF)
#         if key == 'q':
#             vs.cleanup()
#             print('Exiting (via q) demo_superglue.py')
#             break
#         elif key == 'n':  # set the current frame as anchor
#             last_data = {k+'0': pred[k+'1'] for k in keys}
#             last_data['image0'] = frame_tensor
#             last_frame = frame
#             last_image_id = (vs.i - 1)
#         elif key in ['e', 'r']:
#             # Increase/decrease keypoint threshold by 10% each keypress.
#             d = 0.1 * (-1 if key == 'e' else 1)
#             matching.superpoint.config['keypoint_threshold'] = min(max(
#                 0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
#             print('\nChanged the keypoint threshold to {:.4f}'.format(
#                 matching.superpoint.config['keypoint_threshold']))
#         elif key in ['d', 'f']:
#             # Increase/decrease match threshold by 0.05 each keypress.
#             d = 0.05 * (-1 if key == 'd' else 1)
#             matching.superglue.config['match_threshold'] = min(max(
#                 0.05, matching.superglue.config['match_threshold']+d), .95)
#             print('\nChanged the match threshold to {:.2f}'.format(
#                 matching.superglue.config['match_threshold']))
#         elif key == 'k':
#             opt.show_keypoints = not opt.show_keypoints

#     timer.update('viz')
#     timer.print()

#     if opt.output_dir is not None:
#         #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
#         stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
#         out_file = str(Path(opt.output_dir, stem + '.png'))
#         print('\nWriting image to {}'.format(out_file))
#         cv2.imwrite(out_file, out)
# with open('court_point.csv', 'w', newline='') as f:
# # 建立 CSV 檔寫入
#     writer = csv.writer(f)

# # 寫入一列資料
#     writer.writerow(['up_left','mid_left','down_left','up_right','mid_right','down_right'])
#     for i,x in enumerate(T_indx):
#             writer.writerow([x[0], x[1], x[2], x[3],x[4],x[5]])

# f.close

# cv2.destroyAllWindows()
# vs.cleanup()
