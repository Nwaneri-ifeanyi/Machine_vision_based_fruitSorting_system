import cv2
import numpy as np
import os
import time
from keras.models import load_model
from pyfirmata import Arduino, util

model = load_model("/Training/orange_classifier.h5")
board = Arduino("COM7") # ensure your Arduino is connected to
servo = board.get_pin('d:9:s') # connect your servo motor to arduino through pin 9
servo.write(0)


thres = 0.55  # Threshold to detect object
source = "http://10.146.94.23:8080/video"
cap = cv2.VideoCapture(source)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []

classFile = 'coco.names'

# GET CURRENT DIRECTORY PATH
myDirectory = os.path.join(os.getcwd(), 'DataCollected')

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

consecutive_frames = 0
prev_prediction = None

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    # print(classIds, bbox)

    if len(classIds) != 0:

        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            # cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
            #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
            #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            if classId == 53 or classId == 55:  # Here's where you can change the class id
                x, y, w, h = box
                frame = img[y:y + h, x:x + w]
                frame = cv2.resize(frame, (150, 150))


                frame = frame.astype("float32") / 255.0  # Normalizing the image
                frame = np.reshape(frame, (150, 150, 3))
                frame = np.expand_dims(frame, axis=0)  # Adding a batch dimension
                preds = model.predict(frame)
                preds = np.where(preds > 0.5, 1, 0)
                type = preds[0][0]
                if type == 1:
                    print("Ripe Orange")
                    consecutive_frames +=1
                    if consecutive_frames == 8:
                        servo.write(90)
                        time.sleep(1)

                        # Stop the servo
                        servo.write(0)
                        time.sleep(1)

                        consecutive_frames = 0
                else:
                    print("Unripe orange")

    cv2.imshow('Output', img)
    cv2.waitKey(1)
