"""
Detection-based tracking pipeline
"""
import datetime
from time import perf_counter
import cv2
import numpy as np


class Pipeline():
    
    def __init__(self, detector, tracker, resize_image_size=(300,300)):
        self.detector = detector
        self.tracker = tracker
        self.resize_image_size = resize_image_size
    
    def forward(self, frame):
        # preprocess
        pp_frame = cv2.resize(frame, self.resize_image_size)
        pp_frame = cv2.cvtColor(pp_frame, cv2.COLOR_BGR2RGB)

        # get boxes 
        t0 = perf_counter()
        results = self.detector.detect(pp_frame)
        print("detect timer: %f" % (perf_counter() - t0))
        print("    results: %s" % results)

        # update trackers
        t0 = perf_counter()
        tracks = self.tracker.track(results)
        print("track timer: %f" % (perf_counter() - t0))
        print("    tracks: %s" % tracks)
        return tracks

    def get_boxes(self):
        self.tracker.get_boxes()
