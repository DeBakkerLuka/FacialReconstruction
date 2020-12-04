from fastapi import FastAPI, File, UploadFile
import random
import socket
import numpy as np
import time
import threading
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image
from skimage import io, transform
from skimage.color import rgb2gray
from fastapi.responses import FileResponse
from io import BytesIO

app = FastAPI()

@app.get('/')
def welcome():
    return 'Welcome to the Swimming pool squad API for our mlops exam'

@app.get('/test')
def test():
    return 'Hello world, this is a test brought to you by Luka'

@app.get('/getLayers')
def weights():
    model = load_model('facialreconstruction.h5')
    count = len(model.layers)
    return count

@app.get('/processNumpy')
def processRandom():
    model = load_model('facialreconstruction.h5')

    randomNum = random.randint(0,10)

    image = open(f'./faces_mlops/{randomNum}.png', 'rb')
    image_read = image.read()

    image = Image.open(image)

    image = np.array(image)
    image = rgb2gray(image)

    im = transform.resize(image,(100,100),mode='constant',anti_aliasing=True)
    predict = model.predict(np.array([im]))[0]

    return "chosen image " + str(randomNum) + " has numpy values " + str(predict)

@app.post('/process', responses={200: {"content": {"image/png": {}}, "description": "Returns a numpy in string or an image"}})
async def processNumpy(file: UploadFile = File(...)):
    contents = await file.read()

    image = np.array(Image.open(BytesIO(contents)))

    model = load_model('facialreconstruction.h5')

    randomNum = random.randint(0,10)

    image = np.array(image)
    image = rgb2gray(image)

    im = transform.resize(image,(100,100),mode='constant',anti_aliasing=True)
    predict = model.predict(np.array([im]))[0]

    rescaled = (255.0 / predict.max() * (predict - predict.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save('result.png')

    return FileResponse("result.png", media_type="image/png")
