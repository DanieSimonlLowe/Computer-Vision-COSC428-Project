import random
import sys
import time
from copy import deepcopy
from enum import Enum
import numpy as np
import cv2
import pyrealsense2 as rs

from classes.distanceFilter import DistanceFilter

alpha = 0.5
assumed_arm_length = 0.92  # based on 35cm average male arm span with a bit added for wigle room.

min_area = 100

kernel = np.ones((3, 3), np.uint8)


class DetectionState(Enum):
    SAFE = 1
    WARNING = 2
    DANGER = 3

"""
    the detected object code where most of my program resides.
"""
class DetectedObject:
    def __init__(self, mask, box, depth_intrin):
        self.id = random.randint(0, sys.maxsize)
        self.box = box
        self.mask = mask
        self.speed = 0
        self._distance = None
        self._contour = None
        self.tracked_contours = []
        self.time = time.time()

        self.trajectories = []

        # self.speed = 0
        self.filter = None
        self.main_dist = 0

        self.depth_intrin = depth_intrin

    """
        initializes the tracked objects that are the whole object or it's parts.
    """
    def init_trajectories(self, frame_rate):
        pos = self.deproject_box(self.box, self.main_dist)
        self.filter = DistanceFilter(pos[2], pos[0], frame_rate)
        for dist, contour in self.tracked_contours:
            box = np.array(cv2.boundingRect(contour))
            pos = self.deproject_box(box, dist)
            cont_filter = DistanceFilter(pos[2], pos[0], frame_rate, alpha=0.5)

            self.trajectories.append((cont_filter, contour))

    """gets the 3d coordinates of the center of some box with it's depth (is the median depth so it not really the 
    center pos)."""
    def deproject_box(self, box, depth):
        x = box[0] + box[2] * 0.5
        y = box[1] + box[3] * 0.5

        return rs.rs2_deproject_pixel_to_point(
            self.depth_intrin, [x, y], depth
        )

    """updates the trajectories of all the parts of the detected object using the current frame segmentation"""
    def calc_trajectories(self, old, frame_rate):
        self.id = old.id
        self.filter = deepcopy(old.filter)
        self.filter.predict()
        pos = self.deproject_box(self.box, self.main_dist)
        self.filter.update(pos[2], pos[0])

        olds = old.trajectories

        for dist, contour in self.tracked_contours:
            min_diff = 1000
            best_filter = None
            best_age = 0
            best_box = None

            box = np.array(cv2.boundingRect(contour))
            for cont_filter, contour2 in olds:
                box2 = np.array(cv2.boundingRect(contour2))

                diff = np.sum((box - box2) ** 2) + 1000 * (dist - cont_filter.get_current()[0]) ** 2

                if diff < min_diff:
                    min_diff = diff
                    best_filter = deepcopy(cont_filter)
                    best_box = box2

            if best_filter is not None:
                pos = self.deproject_box(best_box, dist)
                best_filter.predict()
                best_filter.update(pos[2], pos[0])

                self.trajectories.append((best_filter, contour))
            else:
                pos = self.deproject_box(box, dist)
                cont_filter = DistanceFilter(pos[2], pos[0], frame_rate, alpha=0.5)

                self.trajectories.append((cont_filter, contour))

        # new_speed = (self.distance() - old.distance()) / (self.time - old.time)
        # self.speed = (new_speed + 9 * old.speed) * 0.1

    """
    calculates the distance of an object from the camera.
    :arg image: the depth image that is used to calculate the distance to the object
    :arg scale: used to scale the values got from the depth camera to get them in meters
    :return distance: the distance from the camera in meters
    :raise error: is thrown if the image or scale has not been provided and the distance is not calculated yet
    """
    def distance(self, image=None, scale=None):
        if self._distance is not None:
            return self._distance
        if image is None or scale is None:
            raise Exception("distance has not been calculated yet.")

        # gets the median distance of the image which is in the mask.
        mat = np.where(self.mask, image, np.nan)
        dist = np.nanmedian(mat) * scale
        self.main_dist = dist

        # Converted the image into the correct format is needed so that the edge detection and contours can run
        # scales the image so that 7.5 meters is 255
        scaled_image = image / (7.5 / (255 * scale))
        # apply the mask to the image and sets all values outside the mask as 255
        regions_image = np.where(self.mask, scaled_image, 255).astype(np.uint8)

        dist_min = dist
        for level in range(0, 250, 5):
            thresh = cv2.inRange(regions_image, level, level + 5)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=5)

            contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # gets the area of the contour and checks that it is large enough

                mask_cnt = np.zeros_like(mat)
                mask_cnt = cv2.drawContours(mask_cnt, cnt, -1, 1, -1)

                # gets the median distance of the things inside the contour
                masked_cnt = np.where(mask_cnt, image, np.nan)
                dist_cnt = np.nanmedian(masked_cnt) * scale

                self.tracked_contours.append((dist_cnt, cnt))

                # if this dist is less then the min dist record the dist and the contour
                if dist_cnt < dist_min and dist - dist_cnt < assumed_arm_length:
                    dist_min = dist_cnt
                    self._contour = cnt
                    has_changed = True

            # if has_changed:
            #     break

        # returns the min dist if it makes sense else returns the median dist.
        self._distance = dist_min
        return dist_min

    """is used for sorting the objects"""
    def get_diff(self, old, image=None, scale=None):
        old_dist = old.main_dist
        self.distance(image, scale)

        return np.sum((self.box - old.box) ** 2) + 10 * (old_dist - self.main_dist) ** 2

    def get_screen_pos(self):

        if self._contour is None:
            return [np.average(indices) for indices in np.where(self.mask >= 255)]
        else:
            m = cv2.moments(self._contour)

            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            return cx, cy

    """
    gets the warning state based upon the distance to the camera
    """

    def get_warning_state(self, image, scale):
        dist = self.distance(image, scale)
        if self._contour is not None:
            box = cv2.boundingRect(self._contour)
        else:
            box = self.box
        pos = self.deproject_box(box, dist)

        if pos[0] ** 2 + pos[2] ** 2 <= 9:
            return DetectionState.DANGER
        for cont_filter, _ in self.trajectories:
            if cont_filter.overlap():
                return DetectionState.WARNING

        if self.filter.overlap():
            return DetectionState.WARNING

        return DetectionState.SAFE

    """
    highlight the image with the provided color on the mask and outlines the section that is closest to the camera if one is detected
    """

    def highlight(self, image, color):
        # this code fills the contours.
        # if len(self.images) > 0:
        #     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),]
        #     for i, img in enumerate(self.images):
        #         if img.shape != (480, 640):
        #             continue
        #         img = np.expand_dims(img, 0).repeat(3, axis=0)
        #
        #         img = np.moveaxis(img, 0, -1)
        #         masked = np.ma.MaskedArray(image, mask=img, fill_value=colors[i % 3])
        #
        #         image_overlay = masked.filled()
        #         image = cv2.addWeighted(image, 0, image_overlay, 1, 0)
        #     return image

        colored_mask = np.expand_dims(self.mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)

        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        if self._contour is not None:
            image_combined = cv2.drawContours(image_combined, self._contour, -1, (0, 0, 0), 3)

        return image_combined

    """
    shows the distance from the camera on the image.
    """

    def show_distance(self, image):
        image = cv2.putText(image, str(self._distance), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2, cv2.LINE_AA)
        return image

    def keep(self, now):
        age = now - self.time
        if age < 2:
            return True
