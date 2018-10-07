import yaml
import logging

from mvnc import mvncapi
import numpy as np
import cv2

from src.box_utils import clip_box, get_box_size
logger = logging.getLogger(__name__)

class Detector():

    def __init__(self, config_path, device):
        """Base Detector class for Movidius
        """
        config = yaml.load(open(config_path))
        self.graph_name = config['graph_name']
        self.graph_path = config['graph_path']
        self.input_size = config['input_size']
        self.score_threshold = config['score_threshold']
        self.box_size_threshold = config['box_size_threshold']

        self._setup(device)

    def _setup(self, device):
        graph = mvncapi.Graph(self.graph_name)
        graph_buffer = read_graph_buffer(self.graph_path)

        fifo_in, fifo_out = graph.allocate_with_fifos(device, graph_buffer, 
            input_fifo_data_type=mvncapi.FifoDataType.FP16,
            output_fifo_data_type=mvncapi.FifoDataType.FP16)

        self.graph = graph
        self.fifo_in = fifo_in
        self.fifo_out = fifo_out

    def _preprocess(self, image):
        """resize images for input tensor and normalize to [-1, 1]"""
        return normalize(image, self.input_size)
        
    def detect(self, image):
        resized_image = self._preprocess(image)
        input_tensor = resized_image.astype(np.float16)
        # send to NCS
        self.graph.queue_inference_with_fifo_elem(
            self.fifo_in, self.fifo_out, input_tensor, None)
        # get result
        output, _ = self.fifo_out.read_elem()
        results = read_output(output)
        # filter 
        results = self._postprocess(results)
        return results

    def _postprocess(self, results):
        """Filter results with low confidence and small boxes"""
        def pass_filter(r):
            pass_score_filter = r['score'] >= self.score_threshold
            pass_size_filter = get_box_size(r['box']) >= self.box_size_threshold
            return pass_score_filter and pass_size_filter 
        
        filtered = [r for r in results if pass_filter(r)]
        return filtered

    def close(self):
        self.fifo_in.destroy()
        self.fifo_out.destroy()
        self.graph.destroy()



##
## Util functions
##

def read_graph_buffer(graph_file_path):
    with open(graph_file_path, mode='rb') as f:
        graph_buffer = f.read()
    return graph_buffer

def normalize(image, dim):
    height, width = dim
    img = cv2.resize(image, (height, width))
    # adjust values to range between -1.0 and + 1.0
    img = img - 127.5
    img = img * 0.007843
    return img

def get_boxes(results):
    """get box from list of dicts"""
    return [r['box'] for r in results]

def read_output(output):
    """extract valid boxes from NCS output"""
    num_valid_boxes = int(output[0])
    results = []
    for box_index in range(num_valid_boxes):
        base_index = 7+ box_index * 7
        if (not np.isfinite(output[base_index]) or
                not np.isfinite(output[base_index + 1]) or
                not np.isfinite(output[base_index + 2]) or
                not np.isfinite(output[base_index + 3]) or
                not np.isfinite(output[base_index + 4]) or
                not np.isfinite(output[base_index + 5]) or
                not np.isfinite(output[base_index + 6])):
            # boxes with non infinite (inf, nan, etc) numbers must be ignored
            logger.warning('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
            continue

        # class_id
        class_id = int(output[base_index + 1])
        # score
        score = output[base_index + 2]
        # box in x1, y1, x2, y2 format
        box = output[base_index + 3], output[base_index + 4], output[base_index + 5], output[base_index + 6]
        box = clip_box(box)
        # result
        result = {'class_id': class_id, 'score': score, 'box': box}
        results.append(result)

    return results