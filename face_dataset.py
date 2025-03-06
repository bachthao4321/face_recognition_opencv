import numpy as np 
import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

face_id = input('Enter user id end press : ')
print('\n Initializing face capture. Look the camera and wait ...')

count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30,30)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (255,0,0), 2)
        count += 1
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", roi_gray)
        cv2.imshow('video',frame)

    k = cv2.waitKey(30)
    if k == 27:
        break
    elif count >= 30:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()