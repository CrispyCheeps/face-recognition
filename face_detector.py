import cv2
from random import randrange

#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
img = cv2.imread('RDJ.png')

#Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)

#Draw rectangles around the faces:
# img, upper left xy axis, bottom right xy axis (u need to add it with the 1st pt), 
# rgb color, thickness of the line

#but if you want to make it dinamically u need to change like down below
i = 0
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)),2)
    i = i+1

print("Total orang yang ada di gambar adalah" , i)

cv2.imshow('Clever Programmer face detector', img)
cv2.waitKey()

print("code completed")