# guess how large the face is from the camera

import cv2

# initialise camera
cap = cv2.VideoCapture(0)

# initialise face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# loop until we have a face
while True:
    # read frame from camera
    ret, frame = cap.read()

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # label the faces in order of size starting from 0 for the largest
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, str(w), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # display the resulting frame
    cv2.imshow('frame', frame)

    # quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break