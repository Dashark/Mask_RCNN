import cv2
import time
from messager import fetch_request, push_response


if __name__ == '__main__':
    init_mask_rcnn()   # 按照缺省参数初始化，不在循环内
    while True:
        image = fetch_request()
        if image is not None:
            test_image(image)  # 分割图像
            # 结果需要放回Redis
            cv2.imshow('py', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('image is None.')
            time.sleep(1)
