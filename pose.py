import mediapipe as mp
import cv2
import time
import pandas as pd

cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0

lm_list = []
lm1 = []

no_of_frame = 50
label = 'HANDSWING'

mpPose = mp.solutions.pose
Pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
def make_data(reult):
    cm = []
    for id, lm in enumerate(result.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        cm.append(lm.x)
        cm.append(lm.y)
        cm.append(lm.z)
        cm.append(lm.visibility)
    return cm

while len(lm_list) <= no_of_frame:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = Pose.process(imgRGB)

    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        lm1 = make_data(result)
        lm_list.append(lm1)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(10) == ord('q'):
        break

df = pd.DataFrame(lm_list)
df.to_csv(label + '.txt')

cap.release()
cv2.destroyAllWindows()
