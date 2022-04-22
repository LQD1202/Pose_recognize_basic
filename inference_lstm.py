import threading

import mediapipe as mp
import cv2
import time

import numpy as np
import pandas as pd
import tensorflow as tf

cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0

lm_list = []
lm1 = []
label = "Warmup..."

mpPose = mp.solutions.pose
Pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
n_time_steps = 10

model = tf.keras.models.load_model("model.h5")
i = 0
warmup_frames = 60
def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    if results[0][0] > 0.5:
        label = "SWING BODY"
    else:
        label = "SWING HAND"
    return label

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        print("Start detect....")

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                lm1.append(lm.x)
                lm1.append(lm.y)
                lm1.append(lm.z)
                lm1.append(lm.visibility)
            lm_list.append(lm1)
            if len(lm_list) == n_time_steps:
                t1 = threading.Thread(target=detect, args=(model, lm_list))
                t1.start()
                lm_list = []

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
