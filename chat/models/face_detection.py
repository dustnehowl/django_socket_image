import argparse
import imutils
import numpy as np
import cv2
import os
import glob

class FaceDetector:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            print("[얼굴 인식 모델 로딩]")
            face_detector = "chat/templates/data/"
            prototxt = face_detector + "deploy.prototxt"
            weights = face_detector + "res10_300x300_ssd_iter_140000.caffemodel"
            net = cv2.dnn.readNet(prototxt, weights)
            cls.__instance = super().__new__(cls)
            cls.__instance.net = net
        return cls.__instance

    def detect(self, image, crop=False, output_path=None):
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = imutils.resize(image, width=500)
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        # print("[얼굴 인식]")
        self.net.setInput(blob)
        detections = self.net.forward()
        minimum_confidence = 0.5
        number = 0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > minimum_confidence:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(W - 1, endX), min(H - 1, endY))
                
                if crop:
                    face = image[startY:endY, startX:endX]
                    face = cv2.resize(face, (224, 224))
                    # if output_path is not None:
                    #     output_file = os.path.join(output_path, f"{number}.jpg")
                    #     #cv2.imwrite(output_file, face)
                    #     number += 1
                    return face

    def detect_from_file(self, image_file, crop=False, output_path=None):
        image = cv2.imread(image_file)
        return self.detect(image, crop, output_path)
