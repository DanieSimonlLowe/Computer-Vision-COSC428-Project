import cv2
import numpy as np

from ultralytics import YOLO
import torch

from classes.detectedObect import DetectedObject


class PedestrianDetector(object):
    """
    Pedestrian detector
    creates an instance of the DetectedObject class to detect pedestrians
    """

    def __init__(self):
        self.last = []
        self.model = YOLO('yolov8n-seg.pt')

    """
    Detects pedestrians
    :arg image: a color image as numpy array
    :return: a list of detected pedestrians as an array of DetectedObjects
    """

    def detect(self, image_input, depth_image, scale):
        # runs the model and gets how it segregates the image
        image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        results = self.model(image)

        detected = []

        for result in results:
            if result is None:
                continue
            if result.masks is None:
                continue
            if result.boxes is None:
                continue

            masks = result.masks.data
            boxes = result.boxes.data
            clss = boxes[:, 5]

            # gets all the indices of all the masks that have a class of people in the image
            people_indices = torch.where(clss == 0)
            # gets all the masks of all the people in the image
            people_masks = masks[people_indices]

            people_boxes = boxes[people_indices]
            # for each mask adds it to the list of detected masks as a DetectedObject
            for mask, box in zip(people_masks, people_boxes):
                obj = DetectedObject(mask.cpu().numpy(), box.cpu().numpy()[:4])
                min_diff = np.inf
                best = None
                for old in self.last:
                    diff = obj.get_diff(old, depth_image, scale)
                    if diff < min_diff:
                        min_diff = diff
                        best = old
                if best is not None:
                    obj.calc_trajectories(best)
                    self.last.remove(best)
                detected.append(obj)
        self.last = detected

        return detected
