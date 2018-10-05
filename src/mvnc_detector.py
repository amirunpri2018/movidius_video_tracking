from mvnc import mvncapi
import numpy as np
import cv2


class BaseDetector():

    def __init__(self, device):
        """Base Detector class for Movidius
        """
        self.graph_name = None
        self.graph_path = None
        self.input_size = (0, 0)
        self.score_threshold = 0.60
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
        
    def predict(self, image):
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
        """Filter results with low confidence"""
        filtered = [r for r in results if r['score'] >= self.score_threshold]
        return filtered

    def close(self):
        self.fifo_in.destroy()
        self.fifo_out.destroy()
        self.graph.destroy()


##
## Model implementations 
##

class VOCDetector(BaseDetector):

    def __init__(self, device):
        self.graph_path = "./models/voc2012/graph"
        self.graph_name = "voc_ssd_mobilenets_graph"
        self.input_size = (300, 300)
        self.score_threshold = 0.60
        self._setup(device)    
                

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

def clip_box(box):
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(1, x2)
    y2 = min(1, y2)
    return [x1, y1, x2, y2]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


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
            print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
            continue

        # class_id
        class_id = int(output[base_index + 1])
        # score
        score = output[base_index + 2]
        # box in x1, y1, x2, y2 format
        box = output[base_index + 3], output[base_index + 4], output[base_index + 5], output[base_index + 6]
        box = clip_box(box)
        box = xyxy_to_xywh(box)

        # result
        result = {'class_id': class_id, 'score': score, 'box': box}
        results.append(result)

    return results