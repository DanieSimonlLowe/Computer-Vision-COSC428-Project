import numpy as np
import cv2

from classes.pedestrianDetector import PedestrianDetector


cap = cv2.VideoCapture(1)

detector = PedestrianDetector()

while cv2.waitKey(1) < 0:
    ret, frame = cap.read()
    outputs = detector.detect(frame)

    print('Found %s objects.' % len(outputs))
    print('\nEnlisting objects:')
    print([out.shape for out in outputs])

    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()

