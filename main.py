import os
import base64
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import cv2
from src.models import get_datetime, read_image_file, resize_image_file
from src.models import face_detection_opencv, load_model_mtcnn, face_detection_mtcnn

APP_DESC = """Description of App"""
app = FastAPI(title='Face Detection API (with FastAPI)', description=APP_DESC)

@app.get("/")
def read_root():
    """Face Detection API Get Method

    Returns:
        response(Dict): Some information about the API
    """
    tarih_saat, _, _ = get_datetime()
    # return {"Hello": "World"}
    return {'DateTime': tarih_saat,
    'Message': 'Face Detection API (with FastAPI)',
    'Department': '',
    'Contact': ''}

class Analyzer(BaseModel):
    filename: str
    img_dimensions: str
    encoded_img: str

@app.post("/facedetection")
# @app.post("/facedetection", response_model=Analyzer)
async def upload_file(file: UploadFile = File(...)):
    tarih_saat, _, _ = get_datetime()
    file_desc = os.path.splitext(file.filename)
    file_extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "heic")
    if not file_extension:
        # return "Image must be *.jpg/jpeg and *.png or *.heic file format!"
        return {'DateTime': tarih_saat,
        'Message': 'Image must be *.jpg/jpeg and *.png or *.heic file format!',
        'File': file.filename,
        'FileName': file_desc[0],
        'FileExt:': file_desc[1]
        }

    # Read Image
    img_object = read_image_file(file=await file.read())
    height, width, channels = img_object.shape
    _, encoded_img = cv2.imencode(file_desc[1], img_object)
    img_encoded = base64.b64encode(encoded_img)

    # Resize Image
    img_object = resize_image_file(image=img_object, width=width, height=height)

    detecter_type = 'mtcnn'
    if detecter_type == 'mtcnn':
        img_object = cv2.cvtColor(img_object, cv2.COLOR_BGR2RGB)
        face_count, face_condifence, face_boxes = face_detection_mtcnn(detector=detector,
         img=img_object)
        return{
            'File': file.filename,
            'FileName': file_desc[0],
            'FileType:': file_desc[1],
            'OriginalDimensions': str((height, width, channels)),
            'ResizeDimensions': str(img_object.shape),
            'FaceCount': face_count,
            'FaceCondifence': face_condifence,
            'FaceBoxes': face_boxes
            # 'Base64Encoded': img_encoded,
        }
    elif detecter_type == 'opencv':
        face_count, face_boxes = face_detection_opencv(img=img_object)
        return{
            'File': file.filename,
            'FileName': file_desc[0],
            'FileType:': file_desc[1],
            'OriginalDimensions': str((height, width, channels)),
            'ResizeDimensions': str(img_object.shape),
            'FaceCount': face_count,
            'FaceBoxes': face_boxes
            # 'Base64Encoded': img_encoded,
        }
    else:
        pass

if __name__ == "__main__":
    detector = load_model_mtcnn()
    uvicorn.run(app, port=8000, host='0.0.0.0')
