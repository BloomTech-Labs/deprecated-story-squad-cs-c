# pull official base image
FROM python:3.8-slim-buster

# set working directory
WORKDIR /usr/src/app

# install python dependencies
RUN python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
RUN apt-get -y install tesseract-ocr
RUN pip install pillow
RUN pip install pytesseract
RUN pip install opencv-contrib-python
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# add app
COPY . .
