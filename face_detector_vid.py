import cv2
from random import randrange

#Load pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose a video to detect faces in (we gonna use webcam stream)
webcam = cv2.VideoCapture(0)

#Looping tak terbatas pada setiap frames
while True:

    #Read the current frame
    #code dibawah mereturn 2 hal,
    #yg 1st adlah boolean, yg ke2 is the actual image
    successful_frame_read, frame = webcam.read()

    #Convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    #Draw rectangles around the faces
    i = 0
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)),5)
        i = i+1

    print("Jumlah orang diframe terdeteksi ", i, " orang")
        
    cv2.imshow('Face detector ala ishak', frame)
    key = cv2.waitKey(1)

    #to break the infinite loop
    if key==81 or key==113:
        break
