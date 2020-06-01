import cv2
import os
import numpy as np
import faceRecognition as fr


test_img=cv2.imread('C:/Users/mayur/Data Science/Computer Vision/Face/tester_img/Mona.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

faces,faceID=fr.labels_for_training_data('C:/Users/mayur/Data Science/Computer Vision/Face/trainingImages')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')

name={0:"Mayur",1:"Madhu",2:"Manoj",3:"Mona"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>37):
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("Face Detecetion",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
