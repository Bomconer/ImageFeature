from fastapi import FastAPI,Request
import numpy as np
from app.hog import genhog
import base64
import cv2


app = FastAPI()

@app.get("/")
def root():
    return {"message": "This is my api"}

def read64(imgstr):
    datastr = imgstr.split(',')[1]
    nparr = np.fromstring(base64.b64decode(datastr),np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_GRAYSCALE)
    return img

@app.get("/api/gethog")
async def read_str(data:Request):
    json = await data.json()
    imgstr = json["img"]
    img = read64(imgstr)
    hog = genhog(img)
    return hog.tolist()