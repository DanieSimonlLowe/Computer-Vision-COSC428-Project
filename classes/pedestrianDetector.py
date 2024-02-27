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

    def detect(self, image_input):

        image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        results = self.model(image)

        detected = []
        for result in results:
            masks = result.masks.data
            boxes = result.boxes.data
            clss = boxes[:, 5]
            people_indices = torch.where(clss == 0)
            people_masks = masks[people_indices]
            for mask in people_masks:
                detected.append(DetectedObject(mask.cpu().numpy()))

        return detected
