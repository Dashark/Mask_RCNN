"""
Mask R-CNN
Train on the toy My dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 my.py train --dataset=/path/to/my/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 my.py train --dataset=/path/to/my/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 my.py train --dataset=/path/to/my/dataset --weights=imagenet

    # Apply color splash to an image
    python3 my.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 my.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json, math
import datetime
from datetime import datetime
import numpy as np
import skimage.draw
import time
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import colorsys
import random
from io import BytesIO

# Root directory of the project
# ROOT_DIR = os.path.abspath(".\\")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "logs"


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, points, title="",
                      figsize=(16, 16), ax=None,
                      colors=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # If no axis is passed, create one and automatically call show()
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        # _, ax = plt.subplots(1)

    # Generate random colors
    colors = colors or random_colors(7)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    # masked_image = image.astype(np.uint32).copy()
    # print(points)
    rpts = points.T
    ax.plot(rpts[0], rpts[1], 'go-')
    ax.imshow(image.astype(np.uint8))
    plt.savefig('test.jpg')
    figdata = BytesIO()
    plt.savefig(figdata, format='png')
    return figdata.getvalue()

def binomial_fitting(boxes, masks, class_ids, class_names):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    v1 = []
    for i in range(N):

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        # Mask
        mask = masks[:, :, i]

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if class_names[class_ids[i]] != 'fish_head':
            continue
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            # print(verts)
            ti = np.argmin(verts, axis=0)
            print(ti)
            v = verts[ti[1]:,:] # 分割的边界
            # 我需要通过拟合得到一个点的集合，XY需要交换一下
            v1 = v[:, [1,0]]
            # 均方差集合
            MSEs = np.array([])
            coefs = [] # 拟合的集合
            for i in range(1, 5): 
                coef = np.polyfit(v1[:,0], v1[:, 1], i) # 拟合
                coefs.append(coef)
                x_fit = np.polyval(coef, v1[:, 0]) # 拟合后的值
                # print(len(v1), len(x_fit))
                # 原值与拟合值的均方差
                MSE = np.linalg.norm(x_fit - v[:, 0], ord=2)**2/len(v)
                MSEs = np.append(MSEs, MSE)
            diffMSE = np.diff(MSEs) # 拟合的均方差的差异
            co_ind = np.where(abs(diffMSE) < 1.0)  # 拟合的MSE差异中选择一个 < 1.0 的
            fx = np.poly1d(coefs[co_ind[0][0]]) # 选择一个合适的
            dfx = fx.deriv()   # 一阶导
            ddfx = dfx.deriv() # 二阶导
            # 产生一个均匀的数据集
            v11 = np.arange(v1[np.argmin(v1, axis=0)[0]][0], v1[np.argmax(v1, axis=0)[0]][0])
            v12 = np.polyval(coefs[co_ind[0][0]], v11) # 计算拟合多项式的值
            r = abs(ddfx(v11))/(1 + dfx(v11)**2)**(3.0/2.0) # 计算密切圆的曲率
            indices = [3]   # 最少3个点为一组子序列
            while (len(r) - indices[-1]) > 3:    # 有足够点分配就循环
                a = np.split(r, indices)  # 分组 r 曲率
                if np.var(a[-2]) < 0.000001:   # 方差够小
                    indices[-1] += 1          # 增加子序列数量
                elif np.var(a[-1]) < -0.001:  # 最后子序列方差够小则结束循环
                    print(a[-1])
                    print(np.var(a[-1]))
                    break
                else:
                    indices = np.append(indices, indices[-1]+3)  # 添加一个子序列
            indices = np.insert(indices, 0, 0)
            x_fit1 = np.array([])
            next = indices
            while len(next) >= 2:
                one, _ = np.split(next, [2])
                coef = np.polyfit(v1[one, 0], v1[one, 1], 1) # 拟合一项式
                x_fit1 = np.append(x_fit1, np.polyval(coef, v11[one[0]:one[1]]))
                print(one, len(x_fit1))
                # print(x_fit1, x_fit[indices[0]:indices[1]+1])
                # print(v[0:20, :])
                _, next = np.split(next, [1])
            # 计算结果 MSE
            MSE = np.linalg.norm(x_fit1 - v12[:indices[-1]], ord=2)**2/len(x_fit1)
            coef = np.polyfit(v1[:,0], v1[:, 1], 2)
            x_fit = np.polyval(coef, v1[:, 0])
            # 合并成一个集合
            v1[:, 1] = x_fit
            v1[:, [0, 1]] = v1[:, [1, 0]]   # 拟合的边界
            # print(v)
            # np.savetxt(result_path+'.txt', v)
    #     plt.show()
    return np.array([v12[indices], v11[indices]]).T, MSE


############################################################
#  代码要移动到Redis
############################################################

# 初始化MaskRCNN环境
def init_mask_rcnn(config):
    import argparse
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for Fish Head.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/my/dataset/",
                        help='Directory of the My dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--label', required=False,
                        metavar="path or label",
                        help='label to classes')
    parser.add_argument('--imageslist', required=False,
                        metavar="path imageslist",
                        help='label to classes')
    args = parser.parse_args()


    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "mymask":
        weights_path = model.find_last() #""
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "mymask":
        #pass
        model.load_weights(weights_path, by_name=True)
        model.load_weights(weights_path, by_name=True)
    else:
        model.load_weights(weights_path, by_name=True)

    return model, args.image
