import time

import numpy as np
import cv2

from classes.detectedObect import DetectionState
from classes.pedestrianDetector import PedestrianDetector
import pyrealsense2 as rs

GREEN = (0, 255, 0)
ORANGE = (255, 0, 255)
RED = (0, 0, 200)

# set up video input for camera
config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
# config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_device_from_file("video.bag", repeat_playback=False)
pipeline = rs.pipeline()
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

print('test1')
# created PedestrianDetector class instance
detector = PedestrianDetector()
print('test2')

# align the color and depth images
align = rs.align(rs.stream.color)

device = profile.get_device()
playback = device.as_playback()
# playback.set_real_time(False)
slow_motion_factor = 0.01

comp_times = []
comp_times1 = []

start_time = time.time()
while cv2.waitKey(1) < 0:
    try:
        # retrieve the depth and color frames
        frames = pipeline.wait_for_frames()
    except RuntimeError:
        break

    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    detector.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    detector.fps = min(depth_frame.get_profile().as_video_stream_profile().fps(),
              color_frame.get_profile().as_video_stream_profile().fps())

    # converted the color frame to a numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # converted the depth frame to a numpy array
    depth_image = np.asanyarray(depth_frame.get_data())

    # detected pedestrians from the color image.
    start = time.time()

    objects = detector.detect(color_image, depth_image, depth_scale)

    min_dist = np.inf
    closest = None

    for obj in objects:
        # for each detected object get there Detection state and highlight them with a color based on that
        state = obj.get_warning_state(depth_image, depth_scale)
        if state == DetectionState.SAFE:
            color_image = obj.highlight(color_image, GREEN)
        elif state == DetectionState.WARNING:
            color_image = obj.highlight(color_image, ORANGE)
        else:
            color_image = obj.highlight(color_image, RED)

        # get the object with the minimum distance
        if min_dist > obj.distance():
            min_dist = obj.distance()
            closest = obj

    comp_times.append(time.time() - start)
    # display the distance of the closest object
    if closest is not None:
        color_image = closest.show_distance(color_image)
    #color_image = cv2.putText(color_image, str(time.time() - start_time), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        #1, (0, 0, 0), 2, cv2.LINE_AA)

    # show the color_image with overlays
    cv2.imshow('frame', color_image)

    # delay = (1 / detector.fps) * slow_motion_factor
    # time.sleep(delay)

print(np.mean(np.array(comp_times)))

pipeline.stop()
cv2.destroyAllWindows()
