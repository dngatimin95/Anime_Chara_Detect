import cv2
import os
import sys
from PIL import Image

width = 100
height = 100

def crop(file_path, cascade = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade):
        raise RuntimeError("%s: not found" % cascade)

    face_cascade = cv2.CascadeClassifier(cascade)

    for file_name in os.listdir(file_path):
        img = cv2.imread(os.path.join(file_path,file_name), cv2.IMREAD_COLOR)
        if img is None:
            os.remove(os.path.join(file_path,file_name))
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            find_faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30, 30))
            for (x, y, w, h) in find_faces:
                crop_img = img[y:y+h, x:x+w]
                resized_img = cv2.resize(crop_img, (width, height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(file_path, file_name), resized_img)

def delete(file_path):
    for char_img in os.listdir(file_path):
        img = Image.open(os.path.join(file_path,char_img))
        w, h = img.size
        img.close()
        if w > width or h > height:
            os.remove(os.path.join(file_path,char_img))

char_file_path = input("Enter the file path to a single character: ")
crop(char_file_path)
delete(char_file_path)
