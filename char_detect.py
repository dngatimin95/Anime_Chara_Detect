import cv2
import os
import numpy as np
from PIL import Image

# Using IP webcam
import urllib.request
import time

url = input("Enter the webcam's IPv4 address (e.g. 123.123.12.1:3000): ") # Insert Ip for webcam
url = "http://" + url + "/shot.jpg"
file_path = input("Enter the file path to the character dataset: ")
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')

def get_char(file_path):
    char_folder_path, char_face = [], []
    ids = set()
    char_id = 1
    char_dic = {}

    for char_folder in os.listdir(file_path):
        char_dic[char_folder] = char_id
        char_folder_path.append(os.path.join(file_path,char_folder))
        char_id += 1

    for char_imgs in char_folder_path:
        all_img_paths = [os.path.join(char_imgs, img) for img in os.listdir(char_imgs)]
        for img_path in all_img_paths:
            char_name = os.path.basename(os.path.dirname(img_path))
            id = char_dic[char_name]
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30, 30))
                for (x, y, w, h) in faces:
                    crop_img = gray[y:y+h, x:x+w]
                    resized_image = cv2.resize(crop_img, (100, 100), interpolation=cv2.INTER_AREA)
                    char_face.append(resized_image)
                    print(id)
                    ids.append(id)
                    # SHOULD BE BETTER TO SCRAPE AND CROP TO LIMIT SIZE OF FILES
    return char_face, ids, char_dic


print ("\n [INFO] Training faces. This will take a few seconds. Wait...")
faces, ids, char_dic = get_char(file_path)
recognizer.train(faces, np.array(ids))
recognizer.save('trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

recognizer.read('trainer.yml')
font = cv2.FONT_HERSHEY_DUPLEX
#capture = cv2.VideoCapture(0) #Uncomment if using computer camera

#id = 0
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
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))#int(minW), int(minH)))

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
