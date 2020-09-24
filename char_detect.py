import cv2
import numpy as np
from PIL import Image
import hashlib
import os

import urllib.request
import time
url = 'http://10.0.1.107:8080/shot.jpg'

path = "char"
recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')

def getImagesAndLabels(path):
    charFolderPath = []
    idCount = 1
    charId = {}
    faceSamples=[]
    ids = []
    for charFiles in os.listdir(path):
        charId[charFiles] = idCount
        charFolderPath.append(os.path.join(path,charFiles))
        idCount += 1

    for charImages in charFolderPath:
        imagePaths = [os.path.join(charImages,img) for img in os.listdir(charImages)]
        print(imagePaths)
        for imagePath in imagePaths:
            print(imagePath) #REMOVE REDUNCY

            charName = os.path.basename(os.path.dirname(imagePath))
            id = charId[charName]
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30, 30))
                for (x, y, w, h) in faces:
                    crop_img = gray[y:y+h, x:x+w]
                    resized_image = cv2.resize(crop_img, (100, 100), interpolation=cv2.INTER_AREA)
                    faceSamples.append(resized_image)
                    print(id)
                    ids.append(id)
                    # SHOULD BE BETTER TO SCRAPE AND CROP TO LIMIT SIZE OF FILES

            #PIL_img = Image.open(imagePath).convert('L') # grayscale
            #img_numpy = np.array(PIL_img,'uint8')
            #charName = os.path.basename(os.path.dirname(imagePath))
            #id = charId[charName]
            #faces = faceCascade.detectMultiScale(img_numpy)
            #for (x,y,w,h) in faces:
            #    faceSamples.append(img_numpy[y:y+h,x:x+w])
            #    ids.append(id)
    print(ids)
    return faceSamples, ids, charId

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids, charId = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.save('trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

recognizer.read('trainer.yml')
font = cv2.FONT_HERSHEY_SIMPLEX
#capture = cv2.VideoCapture(0)

id = 0
names = list(charId.keys())
names.insert(0, "None")
print(names)

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
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
capture.release()
cv2.destroyAllWindows()
