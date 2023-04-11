import cv2
import json
import os
import subprocess
import time


class BarcodeSubprocess:
    def __init__(self, sub_cmd='./barcode', data_dir='docker_tmpfs'):
        self.sub_cmd = sub_cmd
        self.p = None
        self.ticks = None
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def __call__(self, img_data):
        # print('BarcodeCall.__call__')
        self.ticks = int(time.time() * 1e3)
        img_file = os.path.join(self.data_dir, '{}.jpg'.format(self.ticks))
        print(img_file)
        cv2.imwrite(img_file, img_data)
        args = [self.sub_cmd, img_file]
        self.p = subprocess.Popen(args)

    def get(self, timeout=None):
        result = None
        if self.ticks and self.p:
            try:
                self.p.wait(timeout)
                with open(os.path.join(self.data_dir, '{}.json'.format(self.ticks)), 'r') as f:
                    result = json.load(f)
                    print(result)
            except subprocess.TimeoutExpired as e:
                print('SubprocessError:', e)
        self.p = None
        self.ticks = None
        return result


if __name__ == '__main__':
    barcode_sub = BarcodeSubprocess('./opencv_subprocess')
    g_img_data = cv2.imread('lena.jpg')
    barcode_sub(g_img_data)  # 调用__call__方法
    barcode_sub.wait(30)
