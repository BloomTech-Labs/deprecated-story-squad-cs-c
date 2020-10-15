import numpy as np
import sys, os, setuptools, tokenize
from fastapi import FastAPI, APIRouter, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import pytesseract
import textstat.textstat
import io
import cv2
from pydantic import BaseModel

router = APIRouter()

def read_img(img):
  pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
  text = pytesseract.image_to_string(img)
  return(text)

class ImageType(BaseModel):
  url: str

@router.post('/ocr/') 
def ocr(request: Request, file: bytes = File(...)):
  if request.method == 'POST':
    image_stream = io.BytesIO(file)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # blur
    blur = cv2.GaussianBlur(frame, (3,3), 0)

    # convert to hsv and get saturation channel
    sat = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)[:,:,1]

    # threshold saturation channel
    thresh = cv2.threshold(sat, 50, 255, cv2.THRESH_BINARY)[1]

    # apply morphology close and open to make mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # do OTSU threshold to get circuit image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # write black to otsu image where mask is black
    otsu_result = otsu.copy()
    otsu_result[mask==0] = 0

    # Now we use pytesseract to extract the text
    label = read_img(otsu)
    sample_story = label
    ss = textstat.textstat.flesch_kincaid_grade(sample_story)
    
    # Returns the cleaned text transcribed using tesseract.
    # Characters that are to be ignored in the user's uploaded work after being transcribed.
    ig_chr = [
        "~",
        "£",
        "¥",
        "@",
        "«",
        "%",
        "$",
        "‘",
        '"',
        "+",
        "/",
        "\\",
        "|",
        "[",
        "]",
        "{",
        "}",
        "\n",
        "\f",
        "*",
        "^",
    ]

    # For loop that removes unnecessary characters (listed above) that are irrelevant to the user's work 
    # specifically when calling it out afterwards with the complexity score (textstat).
    for c in ig_chr:
        ss = ss.replace(c, "")
    
    return ss
  
