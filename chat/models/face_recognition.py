import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import requests
import numpy as np
from io import BytesIO


class FaceRecognizer:
    __instance = None

    def init_server(self):
        from attendance.models import Member
        from chat.models.face_detection import FaceDetector
        members = Member.objects.all()
        self.image_file_names = []
        self.vectors = []
        print("=====================")
        print("=     member 수     =")
        print("=====================")
        print(len(members))
        self.memberDict = {}
        for idx, member in enumerate(members):
            print(idx,"까지 완료!!")
            self.memberDict[idx] = member.name
            print("member Name :", member.name)
            member_image_url = member.image.storeFileName
            response = requests.get(member_image_url)
            img = Image.open(BytesIO(response.content))
            img_array = np.array(img)
            newFace = FaceDetector().detect(img_array,crop=True)
            newVec = self.get_vector(newFace)
            self.vectors.append(newVec)
            self.image_file_names.append(member.image.storeFileName)
        print("준비 완료")

    def getMemberDict(self):
        return self.memberDict
    
    def getImage_file_names(self):
        return self.image_file_names
    
    def getVectors(self):
        return self.vectors

    def __new__(cls):
        if cls.__instance is None:
            # EfficientNet 모델 정의
            model = EfficientNet.from_pretrained('efficientnet-b0')
            model.eval()

            # 이미지 전처리 함수 정의
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            cls.__instance = super().__new__(cls)
            cls.__instance.model = model
            cls.__instance.preprocess = preprocess

        return cls.__instance
        

    def get_vector_from_path(self, img_path):
        # 이미지 파일을 Pillow Image 객체로 불러옴
        img = Image.open(img_path)
        print(type(img))
        # 이미지 전처리
        img_tensor = self.preprocess(img)
        # 4D mini-batch를 만들기 위해 unsqueeze
        img_tensor = img_tensor.unsqueeze(0)
        # feature vector 추출
        with torch.no_grad():
            features = self.model.extract_features(img_tensor)
            # global average pooling
            features = nn.AdaptiveAvgPool2d((1, 1))(features)
            features = features.view(features.size(0), -1)
        # 추출된 feature vector 반환
        return features.numpy()
    
    def get_vector(self, img):
        img = Image.fromarray(img)
        # 이미지 전처리
        img_tensor = self.preprocess(img)
        # 4D mini-batch를 만들기 위해 unsqueeze
        img_tensor = img_tensor.unsqueeze(0)
        # feature vector 추출
        with torch.no_grad():
            features = self.model.extract_features(img_tensor)
            # global average pooling
            features = nn.AdaptiveAvgPool2d((1, 1))(features)
            features = features.view(features.size(0), -1)
        # 추출된 feature vector 반환
        return features.numpy()

    def euclidean_distance(self, x1, x2):
        """유클리드 거리 계산"""
        return ((x1 - x2) ** 2).sum()

    def cosine_similarity(self, x1, x2):
        """코사인 유사도 계산"""
        x1 = torch.tensor(x1)
        x2 = torch.tensor(x2)
        x1 = x1.flatten()
        x2 = x2.flatten()
        return x1.dot(x2) / (x1.norm() * x2.norm())
