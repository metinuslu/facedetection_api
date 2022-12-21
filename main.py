"""This project was written for face detection and serving it as an API."""

import os
import base64
import uvicorn
from fastapi import FastAPI, File, UploadFile
import cv2
# import filetype
from src.models import get_datetime, read_image_file, resize_image_file
from src.models import face_detection_opencv, load_model_mtcnn, face_detection_mtcnn

APP_DESC = """Description of App"""
app = FastAPI(title='Face Detection API (with FastAPI)', description=APP_DESC)

# Initialize MTCNN Model
detector_mtcnn = load_model_mtcnn()

@app.get("/")
def read_root():
    """Face Detection API Get Method
    Returns:
        response(Dict): Some information about the API
    """
    tarih_saat, _, _ = get_datetime()
    return {'DateTime': tarih_saat,
    'Message': 'Face Detection API (with FastAPI)',
    'Department': '',
    'Contact': ''}

@app.post("/facedetection")
# @app.post("/facedetection", response_model=FaceDetector)
async def upload_file(file: UploadFile = File(...)):
    """Face Detection API Post Method
    Arguments:
        file                : Upload File
    Returns:
        DateTime            : Returns the DateTime information when the API was called.
        File                : Returns the full name information of the uploaded file.
        FileName            : Returns the file name information of the uploaded file.
        FileType            : Returns the file type information of the uploaded file.
        OriginalDimensions  : Returns 'width, height and channel' information of the uploaded image file.
        ResizeDimensions    : Returns new 'width, height and channel' information of the uploaded image file.
        FaceCount           : Returns the number of faces found in the uploaded image file.
        FaceConfidence      : Returns the probability information for the face found in the uploaded image file.
        FaceBoxes           : Returns the bounding box information for each face found in the uploaded image file.
    """
    tarih_saat, _, _ = get_datetime()
    # file_desc = os.path.splitext(file.filename)
    file_name, file_ext = os.path.splitext(file.filename)
    file_extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not file_extension:
        return {'DateTime': tarih_saat,
        'Status': 'Warning',
        'Message': 'Image must be *.jpg/jpeg and *.png file format!',
        'File': file.filename,
        'FileName': file_name,
        'FileType:': file_ext,
        }

    # file_kind = filetype.guess(file.filename)
    # if not (file_kind.extension in ('jpg', 'jpeg', 'png')):
    #     return {'DateTime': tarih_saat,
    #     'Message': 'Image must be *.jpg/jpeg and *.png file format!',
    #     'File': file.filename,
    #     'FileName': file_name,
    #     'FileType:': file_ext,
    #     'FileType:': file_kind.extension,
    #     'FileMime:': file_kind.mime
    #     }

    # Read Image
    detector_type = 'mtcnn'
    try:
        img_object = read_image_file(file=await file.read(), detector_type=detector_type)
        height, width, channels = img_object.shape
    except AttributeError as arr_err:
        return{'DateTime': tarih_saat,
            'Status': 'Warning',
            'Message': 'AttributeError',
            'Error': arr_err
        }
    except IOError as io_err:
        return{'DateTime': tarih_saat,
            'Status': 'Warning',
            'Message': 'IOError',
            'Error': io_err
        }
    except Exception as exc_err:
        return{'DateTime': tarih_saat,
            'Status': 'Warning',
            'Message': 'Exception',
            'Error': exc_err
        }        

    _, encoded_img = cv2.imencode(file_ext, img_object)
    img_encoded = base64.b64encode(encoded_img)

    # Resize Image
    img_object = resize_image_file(image=img_object, width=width, height=height)
    
    # Face Detection
    if detector_type == 'mtcnn':
        img_object = cv2.cvtColor(img_object, cv2.COLOR_BGR2RGB)
        face_count, face_confidence, face_boxes = face_detection_mtcnn(detector=detector_mtcnn,
        image=img_object)
    elif detector_type == 'opencv':
        face_count, face_confidence, face_boxes = face_detection_opencv(image=img_object)
        # face_count, face_confidence, face_boxes = 123, None, 456, 
    # else:
    #     pass

    return {'DateTime': tarih_saat,
    'Status': 'Success',
    'File': file.filename,
    'FileName': file_name,
    'FileType:': file_ext,
    'OriginalDimensions': str((height, width, channels)),
    'ResizeDimensions': str(img_object.shape),
    'FaceCount': face_count,
    'FaceConfidence': face_confidence,
    'FaceBoxes': face_boxes
    # 'Base64Encoded': img_encoded,
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
