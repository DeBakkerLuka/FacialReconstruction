from fastapi import FastAPI, Body
import random
import socket
import time
import threading
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

app = FastAPI()

@app.get('/')
def welcome():
    return 'Welcome to the Swimming pool squad API for our mlops exam'

@app.get('/test')
def test():
    return 'Hello world, this is a test brought to you by Luka'

@app.get('/getWeights')
def weights():
    model = load_model('facialreconstruction.h5')
    count = len(model.layers)
    return count

@app.get('/processRandom')
def random():
    model = load_model('facialreconstruction.h5')

    randomNum = random.randint(0, 10)

    image = open(f'./faces_mlops/{randomNum}.png', 'rb')
    image_read = image.read()

    image = Image.open(image)

    image = np.array(image)
    image = rgb2gray(image)

    im = transform.resize(image,(100,100),mode='constant',anti_aliasing=True)
    predict = model.predict(np.array([im]))[0]

    return "chosen image " + randomNum + "has numpy values " + predict
