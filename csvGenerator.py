import time

import numpy as np
import cv2
from pandas import DataFrame

from classes.detectedObect import DetectionState
from classes.pedestrianDetector import PedestrianDetector
import pyrealsense2 as rs


def generate(input_file, output_file):
    config = rs.config()
    config.enable_device_from_file(input_file, repeat_playback=False)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    # created PedestrianDetector class instance
    detector = PedestrianDetector()
    # device = profile.get_device()
    # playback = device.as_playback()
    # playback.set_real_time(False)

    # align the color and depth images
    align = rs.align(rs.stream.color)
    print('starting video stream')
    # playback = pipeline.get_active_profile().get_device().as_playback()

    state_changes = []
    state_change_times = []
    state_change_target = []
    old_states = []
    start_time = time.time()
    while True:
        print('doing frame')
        try:
            # retrieve the depth and color frames
            frames = pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            break

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        detector.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        detector.fps = min(depth_frame.get_profile().as_video_stream_profile().fps(),
                           color_frame.get_profile().as_video_stream_profile().fps(), 19.6)

        # converted the color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())
        # converted the depth frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        detector.fps = min(depth_frame.get_profile().as_video_stream_profile().fps(),
                           color_frame.get_profile().as_video_stream_profile().fps())

        # detected pedestrians from the color image.
        objects = detector.detect(color_image, depth_image, depth_scale)

        new_states = []
        for obj in objects:
            # for each detected object get there Detection state and highlight them with a color based on that
            state = obj.get_warning_state(depth_image, depth_scale)
            first = True
            new_states.append((obj.id, state))
            for state_id, old_state in old_states:
                if obj.id == state_id:
                    first = False
                    if state != old_state:
                        state_change_times.append(time.time() - start_time)
                        state_changes.append(state)
                        state_change_target.append(obj.id)
            if first:
                state_change_times.append(time.time() - start_time)
                state_changes.append(state)
                state_change_target.append(obj.id)
        old_states = new_states


    pipeline.stop()

    df = DataFrame({'timestamp': state_change_times, 'id': state_change_target, 'state change': state_changes})

    df.to_csv(output_file, index=False)


generate('ee.bag', 'ac.csv')
