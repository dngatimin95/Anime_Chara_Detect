import cv2
import os
import numpy as np
from PIL import Image
from numpy import asarray

# Using IP webcam
import urllib.request
import time

def get_char(file_path):
    char_face, char_dic, char_id = [], {}, []
    ids = 0

    for char_folder in os.listdir(file_path):
        char_folder_path = os.path.join(file_path,char_folder)
        char_dic[ids] = char_folder
        for char_imgs in os.listdir(char_folder_path):
            image = cv2.imread(os.path.join(char_folder_path, char_imgs), cv2.IMREAD_GRAYSCALE)
            char_face.append(np.array(image, dtype=np.uint8))
            char_id.append(ids)
        ids += 1
    return char_face, np.array(char_id), char_dic

def detect():
    use_webcam = True
    url = input("Enter the webcam's IPv4 address (e.g. 123.123.12.1:3000) to use it, otherwise leave it blank: ") # Insert Ip for webcam
    if url == "":
        use_webcam = False
        capture = cv2.VideoCapture(0)
        # Define min window size to be recognized as a face
        minW, minH = 0.1*capture.get(3), 0.1*capture.get(4)
    else:
        url = "http://" + url + "/shot.jpg"
        minW, minH = 0.1 * 432, 0.1 * 288 #default webcam size
    file_path = input("Enter the file path to the character dataset: ")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')

    print ("\n [INFO] Training faces. This will take a few seconds. Wait...")
    char_face, ids, char_dic = get_char(file_path)
    recognizer.train(char_face, ids)
    recognizer.save('trainer.yml')
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    recognizer.read('trainer.yml')
    font = cv2.FONT_HERSHEY_DUPLEX

    while True:
        if use_webcam:
            ip_cam = urllib.request.urlopen(url)
            img_arr = np.array(bytearray(ip_cam.read()),dtype=np.uint8)
            frame = cv2.imdecode(img_arr,-1)
        else:
            __, frame = capture.read() #Reading from computer webcam

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            id, acc = recognizer.predict(gray[y:y+h,x:x+w])
            if (acc < 100):
                name = char_dic[id]
                acc = "  {0}%".format(round(100 - acc))
            else:
                name = "unknown"
                acc = "  {0}%".format(round(100 - acc))

            cv2.putText(frame, str(name), (x+5,y-5), font, 1, (255,255,255), 1)
            cv2.putText(frame, str(acc), (x+5,y+h-5), font, 1, (0,0,0), 1)

        cv2.imshow('AnimeFaceDetect',frame)
        esc = cv2.waitKey(10) & 0xff
        if esc == 27:
            break

    print("\n [INFO] Exiting Program Now...")
    if not use_webcam:
        capture.release()
    cv2.destroyAllWindows()

detect()
