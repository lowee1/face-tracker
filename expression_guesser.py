# this script finds a face from the camera and tries to guess the expression

import cv2

# initialize the camera
camera = cv2.VideoCapture(0)

# initialize the face detector
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# initialize the facial landmark predictor
predictor = cv2.face.FisherFaceRecognizer_create()
predictor.read('emotion_classifier_model.xml')

# initialize the list of labels
labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# loop over the frames from the video stream
while True:
    ret, frame = camera.read()

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the face detections
    for (x, y, w, h) in rects:
        # extract the face ROI
        face = gray[y:y + h, x:x + w]

        # compensate for brightness
        face = cv2.equalizeHist(face)

        # resize face for prediction
        face = cv2.resize(face, (350, 350))

        # predict the face expression
        label, confidence = predictor.predict(face)

        # draw the label and bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, labels[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # show the frame
    cv2.imshow("Frame", frame)

    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
