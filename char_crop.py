import cv2
import os
import sys

WIDTH = 100
HEIGHT = 100

def detect(folder, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    faceCascade = cv2.CascadeClassifier(cascade_file)

    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_COLOR)
        if image is None:
            os.remove(os.path.join(folder,filename))
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))
            for (x, y, w, h) in faces:
                crop_img = image[y:y+h, x:x+w]
                resized_image = cv2.resize(crop_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(folder, filename), resized_image)


detect("rias_gremory")
