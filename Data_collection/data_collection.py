import cv2  # Import OpenCV library
import pandas as pd
import numpy as np
import os  # Import operating system interface
from datetime import datetime
import time
from keras.models import load_model

# Initialize global variables
global imgList
countFolder = 0
count = 0
imgList = []

# Set directory path for storing collected data
myDirectory = os.path.join(os.getcwd(), 'DataCollected')

# Check if the directory for storing data exists
# Increment countFolder until a new directory can be created
while os.path.exists(os.path.join(myDirectory, f'IMG{str(countFolder)}')):
    countFolder += 1
newPath = myDirectory + "/IMG" + str(countFolder)
os.makedirs(newPath)

# Set the confidence threshold for object detection
thres = 0.55

# Set the source for video capture
source = "http://10.146.94.23:8080/video"
cap = cv2.VideoCapture(source)

# Set the video capture frame size and brightness
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Load the class names for the COCO dataset
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the pre-trained neural network model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Set the input size and scale of the neural network
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# Define a function for saving the detected objects' images
def saveData(img):
    global imgList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    fileName = os.path.join(newPath, f'Image_{timestamp}.png')
    cv2.imwrite(fileName, img)
    imgList.append(fileName)


# Continuously capture video and detect objects in real-time
while True:
    # Capture a frame from the video source
    success, img = cap.read()

    # Detect objects in the current frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    # If any objects are detected, save their images and draw bounding boxes around them
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Check if the detected object belongs to a specific class (classId 53 or 55)
            if classId == 53 or classId == 55:
                # Extract the region of interest (ROI) containing the detected object
                x, y, w, h = box
                frame = img[y:y + h, x:x + w]
                frame = cv2.resize(frame, (250, 250))
                # Save the image of the detected object
                saveData(frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
