""" This Module includes the basic and model functions """

from datetime import datetime
import numpy as np
import cv2
from mtcnn import MTCNN
import pytz

def get_datetime():
    """
    Returns the DateTime, Date, and Time information using this function.
        Returns:
            datetime (str)
            date (str)
            time (str)
    """
    tarih_saat = datetime.now(tz=pytz.timezone('Turkey'))
    return tarih_saat.strftime("%Y-%m-%d %H:%M:%S"), \
         tarih_saat.strftime("%Y-%m-%d"), tarih_saat.strftime(" %H:%M:%S")

def read_image_file(file, detector_type='mtcnn'):
    """
    Returns the file received from the API as an image object.
        Parameters:
            file (int):
            type (str):
        Returns:
            img_object (str):
    """
    if detector_type == 'mtcnn':        
        img_np_arr = np.fromstring(file, np.uint8)
        img_object = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    elif detector_type == 'opencv':
        img_object = cv2.imread(file)
    else:
        img_object = None
    return img_object

def resize_image_file(image, width, height):
    """
    Returns the image object by resizing it.
        Parameters:
            image (image array):
            width (int):
            height (int):
        Returns:
            image (str):
    """
    if width*height > 1920*1080:
        scale = 1920 / max(width, height)
        image = cv2.resize(image, (int(width*scale), int(height*scale)))
    return image

def face_detection_opencv(image):
    """
    Returns the face count and face bounding box using OpenCV Haarcascades Model for Face Detection
        Parameters:
            image (image array):
        Returns:
            face_count (int):
            face_boxes (list):
    """
    try:
        face_cascade = cv2.CascadeClassifier('model/opencv/haarcascades/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_boxes = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_confidence = None
        face_count = len(face_boxes)
    except Exception as exc_err:
        print("OpenCv Model:", exc_err)
    return face_count, face_confidence, face_boxes

def load_model_mtcnn():
    """
    Returns the detector from the MTCNN Model for Face Detection
        Returns:
            detector ()
    """
    # detector = MTCNN(select_largest=False, device='cuda')
    detector = MTCNN()
    return detector

def face_detection_mtcnn(detector, image):
    """
    Returns the face count and face confidence and face bounding box using MTCNN Model for Face Detection Model
        Parameters:
            detector ():
            image (image array):
        Returns:
            face_count (int):
            face_confidence (list):
            face_boxes (list):
    """
    face_confidence = []
    face_boxes = []

    faces = detector.detect_faces(image)
    face_count = len(faces)

    if face_count > 1:
        for face in faces:
            face_confidence.append(round(face['confidence'], 3))
            face_boxes.append(face['box'])
    elif face_count == 1:
        face_confidence.append(round(faces[0]['confidence'], 3))
        face_boxes.append(faces[0]['box'])
    else:
        pass
    return face_count, face_confidence, face_boxes
