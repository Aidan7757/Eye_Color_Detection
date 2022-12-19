import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            total_b = 0
            total_g = 0
            total_r = 0
            count = 0
            for x in range(ex, ex+ew):
                for y in range(ey, ey+eh):
                    (b,g,r) = frame[y, x]
                    total_b += b
                    total_g += g
                    total_r += r
                    count += 1
            mean_b = total_b / count
            mean_g = total_g / count
            mean_r = total_r / count
            if mean_b > 220 and mean_g > 220 and mean_r < 50:
                cv2.putText(frame, 'Blue Eyes', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif mean_b < 150 and mean_g < 150 and mean_r > 150:
                cv2.putText(frame, 'Brown Eyes', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif mean_b < 75 and mean_g < 75 and mean_r < 75:
                cv2.putText(frame, 'Black Eyes', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            elif mean_b > 200 and mean_g > 200 and mean_r > 200:
                cv2.putText(frame, 'Gray Eyes', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            elif mean_b > 150 and mean_g > 150 and mean_r < 150 and mean_r > 50:
                cv2.putText(frame, 'Hazel Eyes', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif mean_b < 75 and mean_g > 150 and mean_r > 150:
                cv2.putText(frame, 'Green Eyes', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
            elif mean_b > 150 and mean_g < 75 and mean_r > 150:
                cv2.putText(frame, 'Red Eyes', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()