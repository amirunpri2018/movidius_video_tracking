"""
Video object detection pipeline
"""
import datetime
from time import perf_counter
import cv2
import numpy as np


def scale_box(box, width, height):
    x, y, w, h = box # coco format
    return [x * width, y * height, w * width, h * height]


class Pipeline():
    
    def __init__(self, detector, event_interval=6):
        self.controller = Controller(event_interval=event_interval)    
        self.detector = detector
        self.trackers = []
    
    def detect_and_track(self, frame):
        # preprocess
        pp_frame = cv2.resize(frame, (224, 224))
        pp_frame = cv2.cvtColor(pp_frame, cv2.COLOR_BGR2RGB)

        # get boxes 
        t0 = perf_counter()
        boxes = self.detector.detect(pp_frame)
        print("detect timer: %f" % (perf_counter() - t0))

        # normalize boxes
        width, height, _ = frame.shape
        boxes = [scale_box(box, width, height) for box in boxes]

        # reset timer
        self.controller.reset()

        # get trackers
        self.trackers = [Tracker(frame, box) for box in boxes]

        # return state = True for new boxes
        # if no faces detected, faces will be a tuple.
        new = type(boxes) is not tuple

        return boxes, new
    
    def track(self, frame):
        boxes = [t.update(frame) for t in self.trackers]
        # return state = False for existing boxes only
        return boxes, False
    
    def boxes_for_frame(self, frame):
        if self.controller.trigger():
            return self.detect_and_track(frame)
        else:
            return self.track(frame)



class Controller():
    
    def __init__(self, event_interval=6):
        self.event_interval = event_interval
        self.last_event = datetime.datetime.now()

    def trigger(self):
        """Return True if should trigger event"""
        return self.get_seconds_since() > self.event_interval
    
    def get_seconds_since(self):
        current = datetime.datetime.now()
        seconds = (current - self.last_event).seconds
        return seconds

    def reset(self):
        self.last_event = datetime.datetime.now()


class Tracker():
    
    def __init__(self, frame, box):
        (x,y,w,h) = box
        self.box = (x,y,w,h)
        # Arbitrarily picked KCF tracking
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.box)
    
    def update(self, frame):
        _, self.box = self.tracker.update(frame)
        return self.box


