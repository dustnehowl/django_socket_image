from django.apps import AppConfig
from chat.models.face_detection import FaceDetector
from chat.models.face_recognition import FaceRecognizer


class AttendanceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "attendance"

    def ready(self):
        print("AttendanceConfig입니다.")
        FaceRecognizer().init_server()
