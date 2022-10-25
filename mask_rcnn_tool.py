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

# Root directory of the project
# ROOT_DIR = os.path.abspath(".\\")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = "D:\\Projects\\Mask_RCNN\\mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "logs"

############################################################
#  Configurations
############################################################


class MyConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "mask"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + my

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.5

    IMAGE_RESIZE_MODE = "pad64"
    IMAGE_MIN_DIM = 640  
    IMAGE_MAX_DIM = 1280
    IMAGE_MIN_SCALE = 0

    BACKBONE = "resnet50"

############################################################
#  Dataset
############################################################

class MyDataset(utils.Dataset):

    def print_size(self, poly):
        for p in poly:
            a = np.array(p['all_points_y'])
            height = a.max() - a.min()
            a = np.array(p['all_points_x'])
            width = a.max() - a.min()
            self.areas.append(height * width)
            #if height * width < 4096:
            #    print(width, height)

    def load_my(self, dataset_dir, subset, class_dict):
        """Load a subset of the My dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        self.areas = []
        # Add classes. We have only one class to add.
        for (k, v) in class_dict.items():
            self.add_class("my", v, k)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json"),encoding='UTF-8'))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        print(class_dict)
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            # print(a['regions'])
            print(a['filename'])

            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                class_ids = [class_dict[r['region_attributes']['type']] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                class_ids = [class_dict[r['region_attributes']['type']] for r in a['regions']]
            self.print_size(polygons)
            # print(class_ids)
                        

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "my",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)
        self.areas.sort()
        #print(np.unique(np.round(np.sqrt(self.areas))))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a my dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "my":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        
        class_ids = np.array(info['class_ids'])
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "my":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    class_dict = {}
    if args.label:
        label_file = open(args.label)
        label_lines = label_file.readlines()
        label_id = 1
        for label_line in label_lines:
            label_line = label_line.replace('\n', '')
            class_dict[label_line] = label_id
            label_id = label_id + 1

    # Training dataset.
    dataset_train = MyDataset()
    dataset_train.load_my(args.dataset, "train", class_dict)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MyDataset()
    dataset_val.load_my(args.dataset, "val", class_dict)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1500,
                layers='all')


def display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        class_names, title="", ax=None,
                        show_mask=True, show_box=True,
                        iou_threshold=0.5, score_threshold=0.5):
    """Display ground truth and prediction instances on the same image."""
    # Match predictions to ground truth
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box, gt_class_id, gt_mask,
        pred_box, pred_class_id, pred_score, pred_mask,
        iou_threshold=iou_threshold, score_threshold=score_threshold)
    # # Ground truth = green. Predictions = red
    # colors = [(0, 1, 0, .8)] * len(gt_match)\
    #        + [(1, 0, 0, 1)] * len(pred_match)
    # # Concatenate GT and predictions
    # class_ids = np.concatenate([gt_class_id, pred_class_id])
    # scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    # boxes = np.concatenate([gt_box, pred_box])
    # masks = np.concatenate([gt_mask, pred_mask], axis=-1)
    # # Captions per instance show score/IoU
    # captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
    #     pred_score[i],
    #     (overlaps[i, int(pred_match[i])]
    #         if pred_match[i] > -1 else overlaps[i].max()))
    #         for i in range(len(pred_match))]
    # # Set title if not provided
    # title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    # # Display
    # display_instances(
    #     image,
    #     boxes, masks, class_ids,
    #     class_names, scores, ax=ax,
    #     show_bbox=show_box, show_mask=show_mask,
    #     colors=colors, captions=captions,
    #     title=title)
    return gt_match, pred_match, overlaps

def toSquareBox(bbox):
    """bbox:[y1, x1, y2, x2]
    将它按照宽高比转换为正方形
    并调整左上和右下的坐标
    
    正方形的坐标 [y1, x1, y2, x2]
    """
    box_height = bbox[2] - bbox[0]
    box_width = bbox[3] - bbox[1]
    wh_ratio = box_width / box_height
    box_size = box_width / math.sqrt(wh_ratio)
    y1 = int(bbox[0] + box_height / 2 - box_size / 2)
    y2 = int(y1 + box_size)
    x1 = int(bbox[1] + box_width / 2 - box_size / 2)
    x2 = int(x1 + box_size)
    
    return wh_ratio, box_size, box_height * box_width, [y1, x1, y2, x2]

def recall(model, class_names):
    class_dict = {}
    label_dict = ['background']
    if args.label:
        label_file = open(args.label)
        label_lines = label_file.readlines()
        label_id = 1
        for label_line in label_lines:
            label_line = label_line.replace('\n', '')
            class_dict[label_line] = label_id
            label_dict.append(label_line)
            label_id = label_id + 1

    # Validation dataset
    dataset_val = MyDataset()
    dataset_val.load_my(args.dataset, "val", class_dict)
    dataset_val.prepare()

    pre_correct_dict = {}
    pre_total_dict = {}
    pre_iou_dict = {}
    pre_scores_dict = {}
    gt_total_dict = {}
    for i in range(1, len(class_dict) + 1):
        pre_correct_dict[i] = 0
        pre_total_dict[i] = 0
        pre_iou_dict[i] = 0.0
        pre_scores_dict[i] = 0.0
        gt_total_dict[i] = 0

    backbone_shapes = modellib.compute_backbone_shapes(config, [768,1280,3])
    anchor_boxes = utils.generate_pyramid_anchors(
                config.RPN_ANCHOR_SCALES,
                config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                config.BACKBONE_STRIDES,
                config.RPN_ANCHOR_STRIDE)
    #utils.generate_anchors(300, config.RPN_ANCHOR_RATIOS, [40,40], 32, config.RPN_ANCHOR_STRIDE)
    #print(anchor_boxes)

    rois = []
    obj_groups = []
    # {image_file, [gt_class_id], [gt_box, (y1,x1,y2,x2)], [gt_bbox_area], [gt_wh_ratio], [gt_mask_area], [gt_mask_ratio], [gt_size], }
    for image_id in dataset_val.image_ids:
        print(dataset_val.image_reference(image_id))

        image, image_meta, gt_class_id, gt_box, gt_mask = modellib.load_image_gt(dataset_val, config, image_id, use_mini_mask=False)
        #print(image.shape)
        gt_detects = {}
        gt_detects['image'] = dataset_val.image_reference(image_id)
        gt_detects['gt_class_id'] = gt_class_id
        gt_detects['gt_bbox'] = gt_box
        gt_detects['gt_bbox_area'] = []
        gt_detects['gt_wh_ratio'] = []
        gt_detects['gt_mask_area'] = []
        gt_detects['gt_mask_ratio'] = []
        gt_detects['gt_size'] = []
        for i in range(0, len(gt_class_id)):
            gt_total_dict[gt_class_id[i]] = gt_total_dict[gt_class_id[i]] + 1

            wh_ratio, box_size, box_area, square_box = toSquareBox(gt_box[i])
            mask_area = np.sum(gt_mask[:,:,i]==True)
            mask_ratio = mask_area / box_area
            gt_detects['gt_bbox_area'].append(box_area)
            gt_detects['gt_wh_ratio'].append(wh_ratio)
            gt_detects['gt_mask_area'].append(mask_area)
            gt_detects['gt_mask_ratio'].append(mask_ratio)
            gt_detects['gt_size'].append(box_size)

        molded_image = modellib.mold_image(image, config)
        #print(molded_image.shape)
        # Anchors
        """
        anchors = model.get_anchors(molded_image.shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
        print(anchors)
        # Run object detection
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox =\
            model.keras_model.predict([np.expand_dims(molded_image, 0), np.expand_dims(image_meta, 0), anchors], verbose=0)
        print(detections[0])
        print(mrcnn_class[0])
        print(rpn_class[0])
        """
        #skimage.io.imsave("test.jpg", image)
        start_time = time.time()
        results = model.detect_molded(np.expand_dims(molded_image, 0), np.expand_dims(image_meta, 0), verbose=0)
        end_time = time.time()
        #print("Time: %s" % str(end_time - start_time))
        #print(results)
        r = results[0]
        pre_class_ids = r['class_ids']
        for i in range(0, len(pre_class_ids)):
            pre_total_dict[pre_class_ids[i]] = pre_total_dict[pre_class_ids[i]] + 1
        pre_scores = r['scores']
        #print(r['rois'])
        for roi in r['rois']:
            whr, bsize, _, _ = toSquareBox(roi)
            rois.append([bsize, whr])
            #print(gt_detects['gt_size'])
            #overlaps = utils.compute_iou(roi, gt_detects['gt_bbox'], roi_area, gt_detects['gt_bbox_area'])
            #print(overlaps)
        gt_match, pred_match, overlap = display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        r['rois'], pre_class_ids, pre_scores, r['masks'],
                        class_names, title="", ax=None,
                        show_mask=True, show_box=True,
                        iou_threshold=0.1, score_threshold=0.1)
        gt_detects['rois'] = r['rois']
        gt_detects['gt_match'] = gt_match
        gt_detects['pred_match'] = pred_match
        #print(gt_match)
        """
        visualize.display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        r['rois'], pre_class_ids, pre_scores, r['masks'],
                        class_names, title="", ax=None,
                        show_mask=True, show_box=True,
                        iou_threshold=0.1, score_threshold=0.1)
        """
        for i in range(0, len(pred_match)):
            if pred_match[i] > -1.0:
                #print(r['rois'][i])
                pre_correct_dict[pre_class_ids[i]] = pre_correct_dict[pre_class_ids[i]] + 1
                pre_iou_dict[pre_class_ids[i]] = pre_iou_dict[pre_class_ids[i]] + overlap[i, int(pred_match[i])]
                pre_scores_dict[pre_class_ids[i]] = pre_scores_dict[pre_class_ids[i]] + pre_scores[i]
        obj_groups.append(gt_detects)
    #print(rois)

    print("图片,类别,标注框,标注宽高比,标注尺寸,检测框,检测宽高比,检测尺寸,最大IOU")
    for det in obj_groups:
        for i in range(0, len(det['gt_class_id'])):
            overlaped = utils.compute_overlaps(anchor_boxes, np.reshape(det['gt_bbox'][i],(1,4)))
            omax = max(overlaped)
            #if det['gt_size'][i] > 150 and det['gt_size'][i] < 367:
            if omax[0] > 0.0:
                print(det['image'], end='')
                print(",", label_dict[det['gt_class_id'][i]], ",", det['gt_bbox'][i], ",", det['gt_wh_ratio'][i], ",", det['gt_size'][i], end="")
                if det['gt_match'][i] > -1.0:
                    idx = int(det['gt_match'][i])
                    #print(idx, det['rois'])
                    whr, bsize, _, _ = toSquareBox(det['rois'][idx])
                    print(",", det['rois'][idx], ",", whr, ",", bsize, ",", omax[0])
                else:
                    print(",", 0, ",", 0, ",", 0, ",", omax[0])
        

    tol_pre_correct_dict = 0
    tol_pre_total_dict = 0
    tol_pre_iou_dict = 0
    tol_pre_scores_dict = 0
    tol_gt_total_dict = 0
    
    lines = []
    tile_line = 'Type,Number,Correct,Proposals,Total,Rps/img,Avg IOU,Avg score,Recall,Precision\n'
    lines.append(tile_line)
    for key in class_dict:
        tol_pre_correct_dict = tol_pre_correct_dict + pre_correct_dict[class_dict[key]]
        tol_pre_total_dict = pre_total_dict[class_dict[key]] + tol_pre_total_dict
        tol_pre_iou_dict = pre_iou_dict[class_dict[key]] + tol_pre_iou_dict
        tol_pre_scores_dict = pre_scores_dict[class_dict[key]] + tol_pre_scores_dict
        tol_gt_total_dict = gt_total_dict[class_dict[key]] + tol_gt_total_dict

        type_rps_img = pre_total_dict[class_dict[key]] / len(dataset_val.image_ids)
        if pre_correct_dict[class_dict[key]] > 0:
            type_avg_iou = pre_iou_dict[class_dict[key]] / pre_correct_dict[class_dict[key]]
            type_avg_score = pre_scores_dict[class_dict[key]] / pre_correct_dict[class_dict[key]]
        else:
            type_avg_iou = 0
            type_avg_score = 0

        if gt_total_dict[class_dict[key]] > 0:
            type_recall = pre_total_dict[class_dict[key]] / gt_total_dict[class_dict[key]]
        else:
            type_recall = 0

        if pre_total_dict[class_dict[key]] > 0:
            type_precision = pre_correct_dict[class_dict[key]] / pre_total_dict[class_dict[key]]
        else:
            type_precision = 0
        line = '{:s},{:d},{:d},{:d},{:d},{:.2f},{:.2f}%,{:.2f},{:.2f}%,{:.2f}%\n'.format(key, len(dataset_val.image_ids), pre_correct_dict[class_dict[key]], pre_total_dict[class_dict[key]], gt_total_dict[class_dict[key]], type_rps_img, type_avg_iou * 100, type_avg_score, type_recall * 100, type_precision * 100)
        lines.append(line)
        print(line)

    tol_rps_img = tol_pre_total_dict / len(dataset_val.image_ids)
    if tol_pre_correct_dict > 0:
        tol_avg_iou = tol_pre_iou_dict / tol_pre_correct_dict
        tol_avg_score = tol_pre_scores_dict / tol_pre_correct_dict
    else:
        tol_avg_iou = 0
        tol_avg_score = 0

    if tol_gt_total_dict > 0:
        tol_recall = tol_pre_total_dict / tol_gt_total_dict
    else:
        tol_recall = 0

    if tol_pre_total_dict > 0:
        tol_precision = tol_pre_correct_dict / tol_pre_total_dict
    else:
        tol_precision = 0

    totle_line = '{:s},{:d},{:d},{:d},{:d},{:.2f},{:.2f}%,{:.2f},{:.2f}%,{:.2f}%\n'.format('Total', len(dataset_val.image_ids), tol_pre_correct_dict, tol_pre_total_dict, tol_gt_total_dict, type_rps_img, tol_avg_iou * 100, tol_avg_score, tol_recall * 100, tol_precision * 100)
    print(totle_line)
    lines.append(totle_line)
    
    result_file_name = "result_{:%Y%m%dT%H%M%S}.csv".format(datetime.now())
    result_file = open(result_file_name, 'w+')
    result_file.writelines(lines)
    result_file.close()
    
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, class_names, result_image_path, image_path):
    assert image_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        print(r)
        # Color splash
        # splash = color_splash(image, r['masks'])
        # class_names = ['BG', 'car']
        # masked_image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
        #                     class_names, r['scores'])
        # visualize.display_differences(image, r['rois'], r['masks'], r['class_ids'], 
        #                     class_names, r['scores'])
        # Save output
        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        # skimage.io.imsave(result_image_path, masked_image)
        print("Saved to ", result_image_path)
    # elif video_path:
    #     import cv2
    #     # Video capture
    #     vcapture = cv2.VideoCapture(video_path)
    #     width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = vcapture.get(cv2.CAP_PROP_FPS)

    #     # Define codec and create video writer
    #     file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    #     vwriter = cv2.VideoWriter(file_name,
    #                               cv2.VideoWriter_fourcc(*'MJPG'),
    #                               fps, (width, height))

    #     count = 0
    #     success = True
    #     while success:
    #         print("frame: ", count)
    #         # Read next image
    #         success, image = vcapture.read()
    #         if success:
    #             # OpenCV returns images as BGR, convert to RGB
    #             image = image[..., ::-1]
    #             # Detect objects
    #             r = model.detect([image], verbose=0)[0]
    #             # Color splash
    #             splash = color_splash(image, r['masks'])
    #             # RGB -> BGR to save image to video
    #             splash = splash[..., ::-1]
    #             # Add image to video writer
    #             vwriter.write(splash)
    #             count += 1
    #     vwriter.release()
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == True,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

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

def display_instances(image, boxes, masks, class_ids, class_names, result_path,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=False,
                      colors=None, captions=None):
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

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        # _, ax = plt.subplots(1)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        #ax.text(x1, y1 + 8, caption,
        #        color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask and (class_names[class_ids[i]] == 'fish_head'):
            print(class_names[class_ids[i]])
            maskroll = np.roll(mask, 1, axis=1)
            m = np.logical_xor(mask, maskroll)
            masked_image = apply_mask(masked_image, m, color)

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
            # rb = np.max(verts, axis=0)
            # print(verts[verts[:,1]==rb[1]])
            # rb = np.max(verts[], axis=0)
            # lt = np.min(verts, axis=0)
            ti = np.argmin(verts, axis=0)
            print(ti)
            # v = verts[verts[:,0]>(rb[0]-30)]
            v = verts[ti[1]:,:]
            # 通过拟合得到一个点的集合，XY需要交换一下
            v1 = v[:, [1,0]]
            # 多次拟合得到结果的MSE
            MSEs = np.array([])
            coefs = []
            for i in range(1, 5):
                coef = np.polyfit(v1[:,0], v1[:, 1], i)
                coefs.append(coef)
                x_fit = np.polyval(coef, v1[:, 0])
                # print(len(v1), len(x_fit))
                MSE = np.linalg.norm(x_fit - v[:, 0], ord=2)**2/len(v)
                MSEs = np.append(MSEs, MSE)
            # RMSE = np.linalg.norm(x_fit - v[:, 0], ord=2)/len(v)**0.5
            # MAE = np.linalg.norm(x_fit - v[:, 0], ord=1)/len(v)
            print('MSEs: ', MSEs) #, 'RMSE: ', RMSE, 'MAE: ', MAE)
            print('COEFS: ', coefs)
            diffMSE = np.diff(MSEs)
            print('diffMSE: ', abs(diffMSE))
            co_ind = np.where(abs(diffMSE) < 1.0)  # 拟合的MSE差异中选择一个 < 1.0 的
            print('coef index: ', co_ind)
            fx = np.poly1d(coefs[co_ind[0][0]]) # 选择一个合适的
            dfx = fx.deriv()   # 一阶导
            ddfx = dfx.deriv() # 二阶导
            # TODO v1中元素值不是均匀的，存在r值的跨度
            print(v1, len(v1))
            # print(np.argmax(v1, axis=0)[0])
            # print(np.argmin(v1, axis=0)[0])
            # print(v1[np.argmin(v1, axis=0)[0]][0])
            v11 = np.arange(v1[np.argmin(v1, axis=0)[0]][0], v1[np.argmax(v1, axis=0)[0]][0])
            print(v11, len(v11))
            v12 = np.polyval(coefs[co_ind[0][0]], v11)
            r = abs(ddfx(v11))/(1 + dfx(v11)**2)**(3.0/2.0)
            np.savetxt("r.txt", r)
            print('CUT Circle VAR: ', np.var(r))
            # 一维变二维，数据分组，比如分成20组
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
            # print(v[indices], v1[indices])
            # print(v[0:20, :], x_fit[0:20])
            print('INDICES: ', indices)
            # print(v1[indices[0:2], 1])
            x_fit1 = np.array([])
            next = indices
            while len(next) >= 2:
                one, _ = np.split(next, [2])
                coef = np.polyfit(v1[one, 0], v1[one, 1], 1)
                x_fit1 = np.append(x_fit1, np.polyval(coef, v11[one[0]:one[1]]))
                print(one, len(x_fit1))
                # print(x_fit1, x_fit[indices[0]:indices[1]+1])
                # print(v[0:20, :])
                _, next = np.split(next, [1])
            print('MSE:', np.linalg.norm(x_fit1 - x_fit[:indices[-1]], ord=2)**2/len(x_fit1))
            """
            # 按照 indices 分组
            l = int(len(r) / 10)
            inds = np.arange(l, len(r), l)   # 分组索引
            a = np.split(r, inds)
            # 计算每一组的方差
            var = [np.var(e) for e in a]
            print(var)
            indss = np.append(inds, 0)  # 补充一位对齐
            ind_var = np.array([indss, var]).T  # 转置后配对
            # 方差很小的，合并；方差很大的，就拆分？Numpy二维怎么合并拆分？
            # 当前方差小，则合并后续的一个。移动一个数据。
            # 当前方差大，则拆分一个数据给后续。移动一个数据。
            # 结束条件是什么呢？所有方差在阈值以下？
            for iv in ind_var:
                if iv[1] < threshold:
                    iv[0] += 1
                else:
                    iv[0] -= 1
            """
            # 结束后取每组的首尾作为最终结果
            # 合并成一个集合
            v1[:, 1] = x_fit
            v1[:, [0, 1]] = v1[:, [1, 0]]
            print(v1[0:20])
            co = Polygon(v1, facecolor="none", edgecolor=color)
            ax.add_patch(co)
            # print(v)
            p = Polygon(v, facecolor="none", edgecolor="red")
            np.savetxt(result_path+'.txt', v)
            ax.add_patch(p)
            vv = v1[indices].T
            ax.plot(v12, v11, 'go-')
    ax.imshow(masked_image.astype(np.uint8))
    # ax.savefig('test.png')
    plt.savefig(result_path)
    # ax.clf()
    #plt.close()
    # if auto_show:
    #     plt.show()
    # return ax
    
def display_instances3(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    import cv2 as cv
    n_instances = boxes.shape[0]
    # colors = random_colors(n_instances)

    colors = {1:(1.0,0.0,0.0), 2:(0.0,1.0,0.0), 3:(0.0,0.0,1.0), 4:(0.5,0.0,0.0), 5:(0.0,0.5,0.0), 6:(0.0,0.0,0.5), 7:(0.25,0.0,0.0)}
    colors2 = {1:(255,0,0), 2:(0,255,0), 3:(0,0,255), 4:(128,0,0), 5:(0,128,0), 6:(0,0,128), 7:(64,0,0)}
 
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
    # for i, color in enumerate(boxes):
        if not np.any(boxes[i]):
            continue
 
        # if ids[i] == 0 or  ids[i] == 2 or ids[i] == 3:
        #     continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
 
        image = apply_mask(image, mask, colors[ids[i]])
        image = cv.rectangle(image, (x1, y1), (x2, y2), colors2[ids[i]], 2)
        image = cv.putText(
            image, caption, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, colors2[ids[i]], 2
        )
 
    return image

import glob

def test_image(model, class_names, result_image_path, image_path, config):
    assert image_path

    # Image or video?
    for image_name in glob.glob(image_path):
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_name))
        # Read image
        image = skimage.io.imread(image_name)
        print(image.shape)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # print(r)
        # Color splash
        dt = datetime.now().strftime('%Y%m%d%H%M%S')
        result_image_path = image_name + '_result.png'

        display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, result_image_path, r['scores'])
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
        #skimage.io.imsave(result_image_path, image)
        print("window: (y1, x1, y2, x2)=",window)
        print("scale=",scale)
        print("padding:[(top, bottom), (left, right), (0, 0)]=",padding)
        print("crop=",crop)
        print("Saved to ", result_image_path)

def test_video(model, class_names, result_video_path, video_path):
    assert video_path

    if video_path:
        import cv2 as cv
        # Video capture
        print(video_path)
        vcapture = cv.VideoCapture(video_path)
        width = int(vcapture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv.CAP_PROP_FPS)

        # Define codec and create video writer
        print('width: %d, height: %d, fps: %d' % (width, height, fps))
        vwriter = cv.VideoWriter(result_video_path,
                                  cv.VideoWriter_fourcc(*'DIVX'),
                                  fps, (width, height))

        
        success = True
        frames = []
        frames2 = []
        frame_count = 0

        while success:
            print("frame: ", frame_count)
            # Read next image
            success, frame = vcapture.read()
            if not success:
                break
            
            frame_count += 1
            frames.append(frame)
            # OpenCV returns images as BGR, convert to RGB
            frame2 = frame[..., ::-1]
            frames2.append(frame2)
            
            
            
            # Detect objects
            results = model.detect(frames2, verbose=0)

            for i, item in enumerate(zip(frames, results)):
                frame = item[0]
                r = item[1]
                frame = display_instances3(
                    frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
                )
                name = '{0}.jpg'.format(frame_count + i)
                name = os.path.join('./', name)
                # cv.imwrite(name, frame)
                vwriter.write(frame)
                # print('writing to file:{0}'.format(name))
            # Clear the frames array to start the next batch
            frames = []
            frames2 = []
            # result_image_path = 'tmp.png' 
            # # Color splash
            # display_instances(image, r['rois'], r['masks'], r['class_ids'], 
            #             class_names, result_image_path, r['scores'])
            # new_frame = cv.imread(result_image_path)
            # splash = color_splash(image, r['masks'])
            # splash = splash[..., ::-1]
            # cv.imwrite(result_image_path, splash)
            # RGB -> BGR to save image to video
            
            # Add image to video writer
            # RGB -> BGR to save image to video
            # splash = splash[..., ::-1]
            # Add image to video writer
            # vwriter.write(new_frame)
            # count += 1
            # if count == 80:
            #     break
        vwriter.release()

def test_video2(model, class_names, result_video_path, video_path):
    assert video_path

    if video_path:
        import cv2 as cv
        # Video capture
        vcapture = cv.VideoCapture(video_path)
        width = int(vcapture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv.CAP_PROP_FPS)

        # Define codec and create video writer
        vwriter = cv.VideoWriter(result_video_path,
                                  cv.VideoWriter_fourcc(*'DIVX'),
                                  fps, (1600, 1600))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                result_image_path = 'tmp_frame.jpeg' 
                # Color splash
                display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, result_image_path, r['scores'])
                new_frame = cv.imread(result_image_path)
                # RGB -> BGR to save image to video
                # splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(new_frame)
                count += 1
            # if count == 80:
            #     break
        vwriter.release()
    


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect mys.')
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
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--label', required=False,
                        metavar="path or label",
                        help='label to classes')
    parser.add_argument('--imageslist', required=False,
                        metavar="path imageslist",
                        help='label to classes')
    args = parser.parse_args()

    class_names = ['BG']
    if args.label:
        label_file = open(args.label)
        label_lines = label_file.readlines()
        for label_line in label_lines:
            label_line = label_line.replace('\n', '')
            class_names.append(label_line)

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        class InferenceConfig(MyConfig):
            NUM_CLASSES = len(class_names)
            IMAGES_PER_GPU = 1
            # NUM_CLASSES = 8
            BACKBONE = "resnet50"
            RPN_ANCHOR_SCALES = (5,50,100,200,350)
            RPN_ANCHOR_RATIOS = [0.1,1.0,2.0]
            RPN_ANCHOR_STRIDE = 2
            IMAGE_RESIZE_MODE = "square"
            IMAGE_MIN_DIM = 960
            IMAGE_MAX_DIM = 1280
            IMAGE_MIN_SCALE = 0
            LEARNING_RATE = 0.001

            STEPS_PER_EPOCH = 1000
            RPN_TRAIN_ANCHORS_PER_IMAGE=32
            POST_NMS_ROIS_TRAINING = 300
            POST_NMS_ROIS_INFERENCE = 30
            RPN_NMS_THRESHOLD = 0.5
            USE_MINI_MASK=False
            TRAIN_ROIS_PER_IMAGE = 30
            MAX_GT_INSTANCES = 30

            #FPN_CLASSIF_FC_LAYERS_SIZE = 8
        config = InferenceConfig()
    else:
        class InferenceConfig(MyConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            NUM_CLASSES = len(class_names)
            IMAGES_PER_GPU = 1
            # NUM_CLASSES = 8
            BACKBONE = "resnet50"
            RPN_ANCHOR_SCALES = (5,50,100,200,350)
            RPN_ANCHOR_RATIOS = [0.1,1.0,2.0]
            RPN_ANCHOR_STRIDE = 2
            IMAGE_RESIZE_MODE = "square"
            IMAGE_MIN_DIM = 640
            IMAGE_MAX_DIM = 960
            IMAGE_MIN_SCALE = 0
            LEARNING_RATE = 0.001
            
            RPN_TRAIN_ANCHORS_PER_IMAGE=32
            POST_NMS_ROIS_TRAINING = 300
            POST_NMS_ROIS_INFERENCE = 30
            RPN_NMS_THRESHOLD = 0.5
            USE_MINI_MASK=False
            TRAIN_ROIS_PER_IMAGE = 30
            MAX_GT_INSTANCES = 30
            PRE_NMS_LIMIT = 300

            #FPN_CLASSIF_FC_LAYERS_SIZE = 8
            
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    elif args.weights.lower() == "mymask":
        weights_path = model.find_last() #""
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif args.weights.lower() == "mymask":
        #pass
        model.load_weights(weights_path, by_name=True)
        model.load_weights(weights_path, by_name=True)
    else:
        model.load_weights(weights_path, by_name=True)

    t1=int(round((time.time()) * 1000))
    print("t1:",t1)
    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        dt = datetime.now().strftime('%Y%m%d%H%M%S')
        result_image_path = dt + '_result.png'
        detect_and_color_splash(model, class_names, result_image_path, args.image)
    elif args.command == "recall":
        recall(model, class_names)
    elif args.command == "test_image":
        t2=int(round((time.time())*1000))
        print("t2:",t2)
        dt = datetime.now().strftime('%Y%m%d%H%M%S')
        result_image_path = dt + '_result.png'
        test_image(model, class_names, result_image_path, args.image, config)

        print("t1:",t1)
        print("t2:",t2)
        t3=int(round((time.time()) * 1000))
        print("t3:",t3)
        print("t3-t1=",(t3-t1))
        print("t3-t2=",(t3-t2))
    elif args.command == "test_video":
        dt = datetime.now().strftime('%Y%m%d%H%M%S')
        result_video_path = dt + '_result.avi'
        test_video(model, class_names, result_video_path, args.video)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
