from fastapi import FastAPI, Body
import socket
import time
import threading

app = FastAPI()

@app.get('/')
def welcome():
    return 'Welcome to the Swimming pool squad API for our mlops exam'

@app.get('/test')
def test():
    return 'Hello world, this is a test brought to you by Luka'

