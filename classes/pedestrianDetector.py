import cv2
import numpy as np

from ultralytics import YOLO
import torch
from ultralytics.utils.ops import scale_image

from classes.detectedObect import DetectedObject

detectClasses = {
    0  # 'person'
}


class PedestrianDetector(object):
    def __init__(self):

        self.model = YOLO('yolov8n-seg.pt')
        # self.net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        # ln = self.net.getLayerNames()
        # try:
        #     ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # except IndexError:
        #     ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        #
        # self.ln = ln

    def detect(self, image_input):

        image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        results = self.model(image)

        detected = []
        for result in results:
            masks = result.masks.data
            boxes = result.boxes.data
            # extract classes
            clss = boxes[:, 5]

            # get indices of results where class is 0 (people in COCO)
            people_indices = torch.where(clss == 0)
            # use these indices to extract the relevant masks
            people_masks = masks[people_indices]
            # scale for visualizing results
            people_mask = torch.any(people_masks, dim=0).int()
            detected.append(DetectedObject(people_mask.cpu().numpy()))

        return detected
