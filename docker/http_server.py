import base64
import cv2
import skimage
import json
import time
import numpy as np
from twisted.web import server, resource
from twisted.internet import reactor, endpoints

import fish_head
import parting_line
from barcode_subprocess import BarcodeSubprocess
from mrcnn import utils
from mrcnn.config import Config


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


# Configurations
class InferenceConfig(MyConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 7
    BACKBONE = "resnet50"
    RPN_ANCHOR_SCALES = (5, 50, 100, 200, 350)
    RPN_ANCHOR_RATIOS = [0.1, 1.0, 2.0]
    RPN_ANCHOR_STRIDE = 2
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 960
    IMAGE_MIN_SCALE = 0
    LEARNING_RATE = 0.001

    RPN_TRAIN_ANCHORS_PER_IMAGE = 32
    POST_NMS_ROIS_TRAINING = 300
    POST_NMS_ROIS_INFERENCE = 30
    RPN_NMS_THRESHOLD = 0.5
    USE_MINI_MASK = False
    TRAIN_ROIS_PER_IMAGE = 30
    MAX_GT_INSTANCES = 30
    PRE_NMS_LIMIT = 300
    # FPN_CLASSIF_FC_LAYERS_SIZE = 8


class Counter(resource.Resource):
    isLeaf = True  # important
    # model = None
    # class_names = []

    def __init__(self):
        # self.r = redis.Redis(host='127.0.0.1', port=6379, db=1)
        print('http-server Counter init.')

    def render_GET(self, request):
        print(dir(request))
        request.setHeader(b"content-type", b"text/plain")
        request.setResponseCode(404)
        return b''

    def render_POST(self, request):
        time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        body = request.content.read()  # 获取信息

        # 将base64转Mat
        img_data = json.loads(body.decode())
        decoded = base64.b64decode(img_data['imageData'])
        buf = np.frombuffer(decoded, dtype=np.uint8)
        img = buf.reshape((img_data['imageHeight'], img_data['imageWidth'], -1))
        # im_bgr = img[:, :, [2, 1, 0]]
        fitting_size = img_data['fittingSize']

        request.setHeader(b"content-type", b"application/json")
        if request.uri.rstrip(b'/') != b'/api/maskrcnn':
            request.setResponseCode(404)
            return b''

        barcode_process(img)

        image, window, scale, padding, crop = utils.resize_image(
            img,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
        t1 = int(round((time.time()) * 1000))
        # evaluate
        t2 = int(round((time.time()) * 1000))
        # 重新设计参数
        # Detect objects
        t5 = int(round((time.time()) * 1000))
        r = model.detect([image], verbose=0)[0]
        t6 = int(round((time.time()) * 1000))
        np.save('tmp/{}_mask.npy'.format(time_str), r['masks'])
        skimage.io.imsave('tmp/{}_orig.jpg'.format(time_str), image)

        # 计算鱼头鱼腹分割点
        head_mse, head_samples, belly_mse, belly_samples = parting_line.compute(
            r['masks'], r['class_ids'], class_names, head_points_num=fitting_size, belly_points_num=fitting_size)

        dr_image = parting_line.draw_points(image, belly_samples, head_samples)
        skimage.io.imsave('tmp/{}_dr.jpg'.format(time_str), dr_image)

        h, w, _ = dr_image.shape
        b64 = base64.b64encode(dr_image.tobytes())
        # img_b64 = {'height': h, 'width': w, 'data': b64.decode()}

        data_list1 = {"analysisType": 0, 'mse': head_mse, "dataRegionType": 1,
                      "pointSize": head_samples.shape[0], "pointRegion": [],
                      "associatedFileData": None, "associatedFilePath": None}
        for point in head_samples:
            data_list1["pointRegion"].append({'x': int(point[0]), 'y': int(point[1])})

        data_list2 = {"analysisType": 1, 'mse': belly_mse, "dataRegionType": 1,
                      "pointSize": belly_samples.shape[0], "pointRegion": [],
                      "associatedFileData": None, "associatedFilePath": None}
        for point in belly_samples:
            data_list2["pointRegion"].append({'x': int(point[0]), 'y': int(point[1])})

        barcode_info = barcode_process.get()

        result = {"code": 200, "message": "操作成功",
                  "data": {"associatedFileData": b64.decode(), 'imageHeight': h, 'imageWidth': w,
                           "associatedFilePath": None,
                           "dataSize": 2, "dataList": [data_list1, data_list2]
                           }, 
                   "barcode": barcode_info
                  }

        return json.dumps(result).encode("utf-8")


if __name__ == '__main__':
    class_names = ['BG', 'fish_head', 'fish_eye', 'fish_tail', 'fish_body', 'lateral_fin', 'ventral_fin']
    config = InferenceConfig()
    config.display()
    model, image_name = fish_head.init_mask_rcnn(config)  # 按照缺省参数初始化，不在循环内
    barcode_process = BarcodeSubprocess('./barcode_opencv', './tmpfs')
    # Counter.model = model
    # Counter.class_names = class_names

    endpoints.serverFromString(reactor, "tcp:8011").listen(server.Site(Counter()))
    reactor.run()
