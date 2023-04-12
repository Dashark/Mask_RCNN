import json
import time
import numpy as np
from ctow import pixel_to_world, camera_parameter
from twisted.web import server, resource
from twisted.internet import reactor, endpoints

# 像素坐标转载具坐标
def tray_coordinate(data):
    barcode_pixel = data['barcodePixel']
    barcode_space = data['barcodeSpace']
    bias = data['bias']

    ratio_w = barcode_space['width'] / barcode_pixel['width']
    ratio_h = barcode_space['height'] / barcode_pixel['height']

    img_points = []
    f = camera_parameter["f"]
    c = camera_parameter["c"]
    camera_intrinsic = np.mat(np.zeros((3, 3), dtype=np.float64))
    camera_intrinsic[0, 0] = f[0]
    camera_intrinsic[1, 1] = f[1]
    camera_intrinsic[0, 2] = c[0]
    camera_intrinsic[1, 2] = c[1]
    camera_intrinsic[2, 2] = np.float64(1)
    r = camera_parameter["R"]
    t = np.asmatrix(camera_parameter["T"]).T

    tray_objs = []
    for data_obj in data['dataList']:
        tray_obj = {'analysisType': data_obj['analysisType'], 'dataRegionType': data_obj['dataRegionType']}
        if data_obj['dataRegionType'] == 0 and 'rectRegion' in data_obj:
            data_rect_region = data_obj['rectRegion']
            tray_rect_region = {
                'centerPointX': ratio_w * (data_rect_region['centerPointX'] - barcode_pixel['x']) + bias['x'],
                'centerPointY': ratio_h * (data_rect_region['centerPointY'] - barcode_pixel['y']) + bias['y'],
                'centerPointZ': barcode_space['depth'] + bias['z'],
                'width': ratio_w * data_rect_region['width'],
                'height': ratio_w * data_rect_region['height'],
                'length': -1}
            tray_obj['rectRegion'] = tray_rect_region

        if data_obj['dataRegionType'] == 1 and 'pointRegion' in data_obj:
            tray_obj['pointSize'] = data_obj['pointSize']
            data_point_region = data_obj['pointRegion']
            tray_point_region = []
            for data_point in data_point_region:
                tray_point = {'x': ratio_w * (data_point['x'] - barcode_pixel['x']) + bias['x'],
                              'y': ratio_h * (data_point['y'] - barcode_pixel['y']) + bias['y'],
                              'z': barcode_space['depth'] + bias['z']}
                tray_point_region.append(tray_point)
                img_points.append([data_point['x'], data_point['y']])
            result = pixel_to_world(camera_intrinsic, r, t, img_points)
            for res in result:
                tray_point = {'x': res[0][0], 'y': res[0][1], 'z': res[0][2]}
                tray_point_region.append(tray_point)
            tray_obj['pointRegion'] = tray_point_region

        tray_objs.append(tray_obj)

    return {"code": 200, "message": "操作成功",
            "data": {"dataSize": data['dataSize'], "dataList": tray_objs}}


# 载具坐标转工位坐标
def op_coordinate(data):
    barcode_space = data['barcodeSpace']
    bias = data['bias']

    op_objs = []
    for data_obj in data['dataList']:
        op_obj = {'analysisType': data_obj['analysisType'], 'dataRegionType': data_obj['dataRegionType']}
        if data_obj['dataRegionType'] == 0 and 'rectRegion' in data_obj:
            data_rect_region = data_obj['rectRegion']
            op_rect_region = {'centerPointX': barcode_space['x'] + data_rect_region['centerPointX'] + bias['x'],
                              'centerPointY': barcode_space['y'] + data_rect_region['centerPointY'] + bias['y'],
                              'centerPointZ': barcode_space['z'] + data_rect_region['centerPointZ'] + bias['z'],
                              'height': data_rect_region['height'],
                              'width': data_rect_region['width'],
                              'length': data_rect_region['length']}
            op_obj['rectRegion'] = op_rect_region

        if data_obj['dataRegionType'] == 1 and 'pointRegion' in data_obj:
            op_obj['pointSize'] = data_obj['pointSize']
            data_point_region = data_obj['pointRegion']
            op_point_region = []
            for data_point in data_point_region:
                op_point = {'x': barcode_space['x'] + data_point['x'] + bias['x'],
                            'y': barcode_space['y'] + data_point['y'] + bias['y'],
                            'z': barcode_space['z'] + data_point['z'] + bias['z']}
                op_point_region.append(op_point)
            op_obj['pointRegion'] = op_point_region

        op_objs.append(op_obj)

    return {"code": 200, "message": "操作成功",
            "data": {"dataSize": data['dataSize'], "dataList": op_objs}}


# 工位坐标转加工坐标
def motor_coordinate(data):
    switch_space = data['switchSpace']
    bias = data['bias']

    motor_objs = []
    for data_obj in data['dataList']:
        motor_obj = {'analysisType': data_obj['analysisType'], 'dataRegionType': data_obj['dataRegionType']}
        if data_obj['dataRegionType'] == 0 and 'rectRegion' in data_obj:
            data_rect_region = data_obj['rectRegion']
            op_rect_region = {'centerPointX': switch_space['x'] + data_rect_region['centerPointX'] + bias['x'],
                              'centerPointY': switch_space['y'] + data_rect_region['centerPointY'] + bias['y'],
                              'centerPointZ': switch_space['z'] + data_rect_region['centerPointZ'] + bias['z'],
                              'height': data_rect_region['height'],
                              'width': data_rect_region['width'],
                              'length': data_rect_region['length']}
            motor_obj['rectRegion'] = op_rect_region

        if data_obj['dataRegionType'] == 1 and 'pointRegion' in data_obj:
            motor_obj['pointSize'] = data_obj['pointSize']
            data_point_region = data_obj['pointRegion']
            op_point_region = []
            for data_point in data_point_region:
                op_point = {'x': switch_space['x'] + data_point['x'] + bias['x'],
                            'y': switch_space['y'] + data_point['y'] + bias['y'],
                            'z': switch_space['z'] + data_point['z'] + bias['z']}
                op_point_region.append(op_point)
            motor_obj['pointRegion'] = op_point_region

        motor_objs.append(motor_obj)

    return {"code": 200, "message": "操作成功",
            "data": {"dataSize": data['dataSize'], "dataList": motor_objs}}


class Counter(resource.Resource):
    isLeaf = True  # important

    def __init__(self):
        pass

    def render_GET(self, request):
        print(dir(request))
        request.setHeader(b"content-type", b"text/plain")
        request.setResponseCode(404)
        return b''

    def render_POST(self, request):
        body = request.content.read()  # 获取信息

        request.setHeader(b"content-type", b"application/json")
        try:
            data = json.loads(body.decode())
        except Exception as e:
            print('json loads exception:', e)
            request.setResponseCode(400)
            return b''

        uri = request.uri.rstrip(b'/')
        if uri == b'/api/coordinate/tray':  # 载具坐标
            print('像素坐标转载具坐标')
            result = tray_coordinate(data)
        elif uri == b'/api/coordinate/op':  # 工位坐标
            print('载具坐标转工位坐标')
            result = op_coordinate(data)
        elif uri == b'/api/coordinate/motor':  # 加工坐标
            print('工位坐标转加工坐标')
            result = motor_coordinate(data)
        else:
            result = {"code": 400, "message": "无效的api@{}".format(uri.decode())}

        return json.dumps(result).encode("utf-8")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    endpoints.serverFromString(reactor, "tcp:8012").listen(server.Site(Counter()))
    reactor.run()
