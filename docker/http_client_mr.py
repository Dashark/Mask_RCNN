import sys
import base64
import cv2
import json
import numpy as np
import requests
from skimage import io, transform

if len(sys.argv) != 2:
    print('use {} <xxx.jpg>'.format(sys.argv[0]))
    exit()

# 图像分析接口
img_name = sys.argv[1]
# img = cv2.imread(img_name, cv2.IMREAD_COLOR)
img = io.imread(img_name)

print(img.shape, img.dtype)
height, width, _ = img.shape
img_bytes = img.tobytes()
encoded = base64.b64encode(img_bytes)
print(type(encoded))
img_data = {'imageHeight': height, 'imageWidth': width, 'imageData': encoded.decode(), 'fittingSize': 20}
img_json = json.dumps(img_data)

r1 = requests.post("http://127.0.0.1:8011/api/maskrcnn", data=img_json)
result1 = json.loads(r1.text)
print('图像分析接口:', json.dumps(result1['data']['dataList']))
print('barcode:', result1['barcode'])

dr_img_data = result1['data']['associatedFileData']
decoded = base64.b64decode(dr_img_data)
buf = np.frombuffer(decoded, dtype=np.uint8)
dr_img = buf.reshape((result1['data']['imageHeight'], result1['data']['imageWidth'], -1))
dr_img = dr_img[:, :, [2, 1, 0]]
# cv2.imshow('dr_img', dr_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 载具坐标转化接口
tray_data = {"barcodePixel": {"x": 0, "y": 0, "width": 200, "height": 20},
             "barcodeSpace": {"width": 200, "height": 20, "depth": 20},
             "bias": {"x": 0, "y": 0, "z": 0},
             "dataSize": result1['data']['dataSize'],
             "dataList": result1['data']['dataList']
             }
tray_json = json.dumps(tray_data)
r2 = requests.post("http://127.0.0.1:8012/api/coordinate/tray", data=tray_json)
result2 = json.loads(r2.text)
print('载具坐标转化接口', json.dumps(result2))

# 工位坐标转化接口
op_data = {"stationType": 0,
           "barcodeSpace": {"x": 0, "y": 0, "z": 0, "width": 200, "height": 20, "length": 30},
           "bias": {"x": 0, "y": 0, "z": 0},
           "dataSize": result2['data']['dataSize'],
           "dataList": result2['data']['dataList']
           }
op_json = json.dumps(op_data)
r3 = requests.post("http://127.0.0.1:8012/api/coordinate/op", data=op_json)
result3 = json.loads(r3.text)
print('工位坐标转化接口', json.dumps(result3))

# 加工坐标转化接口
motor_data = {"stationType": 0,
              "switchSpace": {"x": 0, "y": 0, "z": 0, "width": 200, "height": 20},
              "bias": {"x": 0, "y": 0, "z": 0},
              "dataSize": result3['data']['dataSize'],
              "dataList": result3['data']['dataList']
              }
motor_json = json.dumps(motor_data)
r4 = requests.post("http://127.0.0.1:8012/api/coordinate/motor", data=motor_json)
result4 = json.loads(r4.text)
print('加工坐标转化接口', json.dumps(result4))

# print(r.headers, '\n', r.status_code, '\n', r.json, '\n', r.text.encode("utf-8").decode('unicode_escape'))
# print('pts:', result['pts'])
# print('mse:', result['mse'])
# img_data = result['img_data']
# decoded = base64.b64decode(img_data['data'])
# buf = np.frombuffer(decoded, dtype=np.uint8)
# img = buf.reshape((img_data['height'], img_data['width'], -1))
# img = img[:, :, [2, 1, 0]]
# cv2.imshow('py', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# decoded = base64.b64decode(result['img_png'])
# with open("client.png", "wb") as f:
#     f.write(decoded)

