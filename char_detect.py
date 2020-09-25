import cv2
import os
import numpy as np
from PIL import Image
from numpy import asarray

# Using IP webcam
import urllib.request
import time

url = input("Enter the webcam's IPv4 address (e.g. 123.123.12.1:3000): ") # Insert Ip for webcam
url = "http://" + url + "/shot.jpg"
file_path = input("Enter the file path to the character dataset: ")
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')

def get_char(file_path):
    char_folder_path, char_face, ids, char_dic = [], [], [], {}
    char_id = 1

    for char_folder in os.listdir(file_path):
        char_dic[char_folder] = char_id
        char_id += 1
        char_folder_path = os.path.join(file_path,char_folder)
        for char_imgs in os.listdir(char_folder_path):
            img_path = os.path.join(char_folder_path, char_imgs)
            char_name = os.path.basename(char_folder)
            id = char_dic[char_name]
            #image = Image.open(img_path).convert('L')
            #image_arr = asarray(image)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #gray = cv2.equalizeHist(gray)
            char_face.append(np.array(image, dtype=np.uint8))
            ids.append(id)
    return char_face, np.array(ids), char_dic

print ("\n [INFO] Training faces. This will take a few seconds. Wait...")
faces, ids, char_dic = get_char(file_path)
recognizer.train(faces, ids)
recognizer.save('trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

recognizer.read('trainer.yml')
font = cv2.FONT_HERSHEY_DUPLEX
#capture = cv2.VideoCapture(0) #Uncomment if using computer camera

names = list(char_dic.keys())
names.insert(0, "None")

# Define min window size to be recognized as a face
#minW = 0.1*capture.get(3)
#minH = 0.1*capture.get(4)

while True:
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    frame = cv2.imdecode(imgNp,-1)

    #ret, frame = capture.read() #Reading from computer webcam
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))#int(minW), int(minH)))

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # If confidence is less them 100 ==> "0" : perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('AnimeFaceDetect',frame)
    esc = cv2.waitKey(10) & 0xff
    if esc == 27:
        break

print("\n [INFO] Exiting Program Now...")
#capture.release() #Uncomment if using computer camera
cv2.destroyAllWindows()
