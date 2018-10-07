""" Sample Detector using Tensorflow SSD Mobilenet COCO detector
"""
import logging
import cv2
import numpy as np
import tensorflow as tf
from .box_utils import xywh_to_xyxy

logger = logging.getLogger(__name__)

class Detector():

    def __init__(self, model_path):
        # path to frozen inference
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

    def get_tensor_dict(self):
        """Get handles to input and output tensors"""
        ops = self.graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
            ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)
        return tensor_dict
    
    def _setup(self): 
        # TODO
        pass
    
    def _download_model(self):
        # TODO
        pass

    def output(self, image):
        with self.graph.as_default():
            with tf.Session() as sess:
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                tensor_dict = self.get_tensor_dict()
                # run inference
                output_dict = sess.run(tensor_dict,
                    feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

        return output_dict

    def detect(self, image, threshold=0.09):
        """Return detection boxes for image frame"""
        # output_dict
        output = self.output(image)
        if output['num_detections'] < 1:
            return []

        boxes = output['detection_boxes']
        scores = output['detection_scores']
        # filter for scores > threshold
        threshold= 0.10
        boxes = [box for box, score in zip(boxes, scores) if score > threshold]
        return boxes


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


if __name__ == '__main__':
    from src.utils import resize_image, draw_boxes, GREEN, BLUE
    from time import perf_counter

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filepath", "-i", type=str,
        action='store',
        help='Input filepath')
    parser.add_argument("--output_filepath", "-o", type=str,
        action='store',
        help='Output filepath')
    parser.add_argument("--model_path", "-p", type=str,
        action='store',
        help='Path to detector model file')

    args = parser.parse_args()

    # read image
    image = cv2.imread(args.input_filepath)
    # preprocess
    pp_image = cv2.resize(image, (224, 224))
    pp_image = cv2.cvtColor(pp_image, cv2.COLOR_BGR2RGB)
    # detect
    detector = Detector(args.model_path)
    t0 = perf_counter()
    boxes = detector.detect(pp_image)
    logger.info("detect timer: %f" % (perf_counter() - t0))

    # scale boxes back to image size
    def scale_box(box, width, height):
        x, y, w, h = box # coco format
        return [x * width, y * height, w * width, h * height]
    width, height, _ = image.shape
    boxes = [scale_box(box, width, height) for box in boxes]
    # draw
    boxes = [xywh_to_xyxy(box) for box in boxes] # convert to xyxy format
    image = draw_boxes(image, boxes, GREEN)
    # save
    cv2.imwrite(args.output_filepath, image)
    