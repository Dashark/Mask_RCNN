import cv2
import time
from redisdb import messager as redis
import fish_head
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

if __name__ == '__main__':
    class_names = ['BG', 'fish_head', 'fish_eye', 'fish_tail', 'fish_body', 'lateral_fin', 'ventral_fin']
    config = InferenceConfig()
    config.display()
    model = fish_head.init_mask_rcnn(config)   # 按照缺省参数初始化，不在循环内
    while True:
        image = redis.fetch_request()
        if image is not None:
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)
            t1=int(round((time.time()) * 1000))
            # evaluate
            t2=int(round((time.time())*1000))
            # 重新设计参数
            # Detect objects
            t5=int(round((time.time())*1000))
            r = model.detect([image], verbose=0)[0]
            t6=int(round((time.time())*1000))
            print("t6-t5=",(t6-t5))
            pts, mse = fish_head.binomial_fitting(r['rois'], r['masks'], r['class_ids'], class_names)
            print(pts)
            print(mse)
            image = fish_head.display_instances(image, pts)

            t3=int(round((time.time()) * 1000))
            print("t3-t1=",(t3-t1))
            print("t3-t2=",(t3-t2))
            # 结果需要放回Redis
            cv2.imshow("test", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('image is None.')
            time.sleep(1)
