# this script is used to track the face from the camera
# and draw a rectangle around it
import cv2
    
# initialize the camera
cap = cv2.VideoCapture(0)

# create the haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# loop until the camera is working
while True:
    # read the frame
    ret, frame = cap.read()
    
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # loop over the faces
    for (x,y,w,h) in faces:
        # draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    
    # display the frame
    cv2.imshow('frame', frame)
    
    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
