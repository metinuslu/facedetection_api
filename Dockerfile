FROM python:3.10

LABEL version="1.0"
LABEL maintainer=""
LABEL maintainer.mail=""

# Copy the source code
ADD . face_detection_api
#COPY . /face_detection_api
WORKDIR /face_detection_api

# Update System
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Update Pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r env/requirements.txt
#RUN pip install --no-cache-dir -r env/requirements.txt

# Start the application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", 8000]
CMD ["python", "main.py"]