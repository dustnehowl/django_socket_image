from django.apps import AppConfig
from .models.face_recognition import FaceRecognizer
from .models.face_detection import FaceDetector


class ChatConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chat'

    @classmethod
    def ready(cls):
        print("ChatConfig입니다.")
        # FaceRecognizer 모듈 생성
        #print("[얼굴 검출 객체 생성]")
        cls.face_detector = FaceDetector()
        #print("[얼굴 인식 객체 생성]")
        cls.face_recognizer = FaceRecognizer()

    @classmethod
    def get_face_detector(cls):
        return cls.face_detector
    
    @classmethod
    def get_face_recognizer(cls):
        return cls.face_recognizer