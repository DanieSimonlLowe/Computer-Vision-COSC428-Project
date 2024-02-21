import numpy as np
import cv2
from matplotlib import pyplot as plt

detectClasses = {
    0 # 'person'
}

class PedestrianDetector(object):
    def __init__(self):
        self.classes = open('coco.names').read().strip().split('\n')

        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')

        self.net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        ln = self.net.getLayerNames()
        try:
            ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        except IndexError:
            ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.ln = ln

    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)

        detected = []
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                if not classID in detectClasses:
                    continue
                confidence = scores[classID]
                if confidence >= 0.3:
                    detected.append(out)

        return outputs
