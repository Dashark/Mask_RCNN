import cv2
import time
from messager import fetch_request, push_response


if __name__ == '__main__':
    while True:
        image = fetch_request()
        if image is not None:
            cv2.imshow('py', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('image is None.')
            time.sleep(1)
