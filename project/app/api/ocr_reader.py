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
  pytesseract.pytesseract.tesseract_cmd = '/opt/local/bin/tesseract'
  text = pytesseract.image_to_string(img)
  return(text)

class ImageType(BaseModel):
  url: str

@router.post("/imgtr") 
def prediction(request: Request, file: bytes = File(...)):
    if request.method == "POST":
      image_stream = io.BytesIO(file)
      image_stream.seek(0)
      file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
      frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
      label = read_img(frame)
      ss = label
    
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
