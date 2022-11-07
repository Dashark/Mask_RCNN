import cv2
import redis
import numpy as np

request_key = "maskrcnn:request"
response_key = "maskrcnn:response"
r = redis.Redis(host='127.0.0.1', port=6379, db=1)


def fetch_request():
    try:
        img_key = r.rpop(request_key)
        if img_key:
            bytes_decode = r.get(img_key)
            if bytes_decode:
                if bytes_decode and isinstance(bytes_decode, bytes):
                    np_decode = np.frombuffer(bytes_decode, np.uint8)
                    img_decode = cv2.imdecode(np_decode, cv2.IMREAD_COLOR)
                    return img_decode
            else:
                return None
        else:
            return None

    except Exception as e:
        print('fetch_request exception: ', e)
        return None


def push_response(message: str):
    try:
        r.lpush(response_key, message)
        return True
    except Exception as e:
        print('fetch_request exception: ', e)
        return False
