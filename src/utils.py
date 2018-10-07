import cv2
import logging

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

def draw_boxes(frame, boxes, color=GREEN):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame

def resize_image(image, size_limit=500.0):
    max_size = max(image.shape[0], image.shape[1])
    if max_size > size_limit:
        scale = size_limit / max_size
        _img = cv2.resize(image, None, fx=scale, fy=scale)
        return _img
    return image
