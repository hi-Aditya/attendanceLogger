import cv2
import numpy as np
import face_recognition as fcreg
import os
from datetime import datetime

path = r'C:\Users\adim4\Desktop\Python\Images'          #path of folder of images
images = []                                             #list of images
stuNames = []                                           #list of logged students
myList = os.listdir(path)                               #myList contains list of every file in 'path'
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')                 #use formatting to read the image
    images.append(curImg)
    stuNames.append(os.path.splitext(cl)[0])

def imgEncoder(images):                                 #create a list of encoded images from given list of images
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #converting BGR to RGB
        encodes = fcreg.face_encodings(img)[0]          #encode image
        encodeList.append(encodes)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

knownEncodings = imgEncoder(images)

vid = cv2.VideoCapture(0)                               #live video from 0(laptop camera)

while True:
    success, img = vid.read()                           #read frames from webcam
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)         #resize img for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrames = fcreg.face_locations(imgS)         #faces in current frame
    encodeCurFrames = fcreg.face_encodings(imgS)

    for encodeFace,faceLoc in zip(encodeCurFrames, facesCurFrames):
        matches = fcreg.compare_faces(knownEncodings,encodeFace)      #compare current face with ones from list
        faces = fcreg.face_distance(knownEncodings,encodeFace)
        matched = np.argmin(faces)                                    #lowest face distance is possible match

        if matches[matched]:
            name = stuNames[matched].upper()                                              #
            y1,x2,y2,x1 = faceLoc                                                         #coordinates of face detect
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4                                             #resize image to o.g. for rectangle
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y1-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y1-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)  #text after detection
            markAttendance(name)

