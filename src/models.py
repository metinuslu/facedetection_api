""" This Module includes the basic and model functions """

from datetime import datetime
import numpy as np
import cv2
from mtcnn import MTCNN
import pytz

def get_datetime():
    "This functions returns Datetime, Date and Time"
    tarih_saat = datetime.now(tz=pytz.timezone('Turkey'))
    return tarih_saat.strftime("%Y-%m-%d %H:%M:%S"), \
         tarih_saat.strftime("%Y-%m-%d"), tarih_saat.strftime(" %H:%M:%S")

def read_image_file(file):
    "Read from image file"
    img_np_arr = np.fromstring(file, np.uint8)
    img_object = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    return img_object

def resize_image_file(image, width, height):
    "Resize from image file"
    if width*height > 1920*1080:
        scale = 1920 / max(width, height)
        image = cv2.resize(image, (int(width*scale), int(height*scale)))
    return image

def face_detection_opencv(img):
    "Face Detection OpenCV Haaarcascades Model"
    face_cascade = cv2.CascadeClassifier('model/opencv/haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_boxes = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_count = len(face_boxes)
    return face_count, face_boxes

def load_model_mtcnn():
    "Load MTCNN Model"
    # detector = MTCNN(select_largest=False, device='cuda')
    detector = MTCNN()
    return detector

def face_detection_mtcnn(detector, img):
    "Face Detection MTCNN Model"
    face_confidence = []
    face_boxes = []

    faces = detector.detect_faces(img)
    face_count = len(faces)

    if face_count > 1:
        for face in faces:
            face_confidence.append(round(face['confidence'], 3))
            face_boxes.append(face['box'])
    elif face_count == 1:
        face_confidence.append(round(faces[0]['confidence'],3))
        face_boxes.append(faces[0]['box'])
    else:
        pass
    return face_count, face_confidence, face_boxes
