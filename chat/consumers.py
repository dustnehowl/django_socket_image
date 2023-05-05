# chat/consumers.py
import json
import base64
from PIL import Image
import io
from .apps import ChatConfig
from .models.face_detection import FaceDetector
from .models.face_recognition import FaceRecognizer
from django.http import JsonResponse
import numpy as np

from channels.generic.websocket import WebsocketConsumer


def save_base64_image(image_data, file_name):
    with open(file_name, "wb") as fh:
        print(len(base64.b64decode(image_data.encode())))
        fh.write(base64.b64decode(image_data.encode()))


def save_image_from_bytes(bytes_data, path):
    image = Image.open(io.BytesIO(bytes_data))
    image.save(path)


class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.face_detector = ChatConfig.get_face_detector()
        self.face_recognizer = ChatConfig.get_face_recognizer()
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data=None, bytes_data=None):
        if text_data:
            print(text_data)
            self.send(text_data=text_data)
        else:
            image_path = 'chat/templates/data/images/realTimeFace.jpg'
            save_image_from_bytes(bytes_data, image_path)
            newFace = self.face_detector.detect_from_file(image_file=image_path, crop=True)
            newVec = self.face_recognizer.get_vector(newFace)

            res = []
            for idx, vec in enumerate(self.face_recognizer.getVectors()):
                distance = self.face_recognizer.euclidean_distance(newVec, vec)
                res.append([distance, self.face_recognizer.getMemberDict()[idx]])
            res.sort(key=lambda x: x[0])

            for i in range(len(res)):
                res[i][0] = float(res[i][0])

            res_json = {
                'type': 'res',
                'data': res
            }
            # 데이터를 문자열로 변환하여 클라이언트로 보내기
            self.send(text_data=json.dumps(res_json))
