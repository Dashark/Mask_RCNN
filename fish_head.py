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
            rb = np.max(verts, axis=0)
            # print(verts[verts[:,1]==rb[1]])
            # rb = np.max(verts[], axis=0)
            lt = np.min(verts, axis=0)
            ti = np.argmin(verts, axis=0)
            print(ti)
            # v = verts[verts[:,0]>(rb[0]-30)]
            v = verts[ti[1]:,:]
            # 我需要通过拟合得到一个点的集合，XY需要交换一下
            v1 = v[:, [1,0]]
            coef = np.polyfit(v1[:,0], v1[:, 1], 2)
            x_fit = np.polyval(coef, v1[:, 0])
            # 合并成一个集合
            v1[:, 1] = x_fit
            v1[:, [0, 1]] = v1[:, [1, 0]]
            co = Polygon(v1, facecolor="none", edgecolor=colors[0])
            ax.add_patch(co)
            # print(v)
            p = Polygon(v, facecolor="none", edgecolor=color)
            np.savetxt(result_path+'.txt', v)
            ax.add_patch(p)
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

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
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

    t1=int(round((time.time()) * 1000))
    print("t1:",t1)
    # evaluate
    if args.command == "test_image":
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
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
