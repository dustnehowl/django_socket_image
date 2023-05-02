# chat/consumers.py
import json
import base64
from PIL import Image
import io

from channels.generic.websocket import WebsocketConsumer


def save_base64_image(image_data, file_name):
    with open(file_name, "wb") as fh:
        print(len(base64.b64decode(image_data.encode())))
        fh.write(base64.b64decode(image_data.encode()))


def save_image_from_bytes(bytes_data):
    image = Image.open(io.BytesIO(bytes_data))
    image.save('chat/templates/data/images/realTimeFace.jpg')


class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data=None, bytes_data=None):
        if text_data:
            print(text_data)
            self.send(text_data=text_data)
        else:
            save_image_from_bytes(bytes_data)
