import cv2
import sys
import logging
import yaml

from mvnc import mvncapi

from src.utils import resize_image, draw_boxes
from src.box_utils import scale_box
from src.pipeline import Pipeline
from src.mvnc_detector import Detector
from src.iou_tracker import IOUTracker
from src.inference import write_api

logger = logging.getLogger(__name__)

def process(video_in, pipeline, video_out=None, headless=True, inference=True):
    """Run video through pipeline"""
    while video_in.isOpened():
        ok, frame = video_in.read()
        if not ok:
            logger.error('Error reading video')
            break

        # quit
        if not headless:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # main loop
        tracks = pipeline.forward(frame)
        boxes = [track['box'] for track in tracks]
        boxes = [scale_box(box, frame) for box in boxes]
        logger.debug("boxes: %s" % boxes)
        frame = draw_boxes(frame, boxes)

        # inference mode - write data to api
        if inference:
            write_api(tracks)

        # display resulting frame
        if not headless:
            cv2.imshow('Video', frame)

        # Write the frame into output file
        if video_out:
            video_out.write(frame)


def init_device():
    # Get a list of ALL the sticks that are plugged in
    # we need at least one
    device_list = mvncapi.enumerate_devices()
    if len(device_list) == 0:
        logger.error('No devices found')
        quit()

    # Pick the first stick to run the network
    device = mvncapi.Device(device_list[0])

    # Open the NCS
    device.open()
    return device 

def setup_detector(detector_config, device):
    """Setup detector"""
    detector = Detector(detector_config, device)
    return detector

def setup_tracker():
    return IOUTracker()

def run_video(input_filepath, detector_config, output_filepath=None):
    """
    Args:
        input_filepath: input video filepath. Set to 0 for webcam, or other device no.
        detector_config: path to detector config file. 
        output_filepath: filepath to save result video, set to None to disable saving 
                to disk.Default=None.
    """
    if input_filepath == "0":
        input_filepath = 0
    video_capture = cv2.VideoCapture(input_filepath)

    # exit if video not opened
    if not video_capture.isOpened():
        logger.error('Cannot open video')
        sys.exit()
    
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    video_out = None
    if output_filepath is not None:
        video_out = cv2.VideoWriter(output_filepath, 
            cv2.VideoWriter_fourcc(*'MPEG'), 
            20., 
            (frame_width,frame_height))

    # init detector
    device = init_device()
    detector = setup_detector(detector_config, device)

    # init tracker
    tracker = setup_tracker()

    # init detection pipeline
    # TODO: pass image_size config  
    pipeline = Pipeline(detector=detector, tracker=tracker, resize_image_size=(300,300))

    # run processing
    process(video_capture, pipeline, video_out=video_out, headless=False)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

    # shutdown device
    detector.close()
    device.close()
    device.destroy()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filepath", "-i", type=str,
        action='store',
        help='Input filepath')
    parser.add_argument("--config", "-c", type=str, 
        action="store",
        help="Config file path, default=./CONFIG")
    parser.add_argument("--output_filepath", "-o", type=str,
        default=None, 
        action='store',
        help='Output filepath')
    args = parser.parse_args()

    config = yaml.load(open("./CONFIG", "rb"))
    detector_config = config.get("detector_config")

    run_video(args.input_filepath, detector_config, output_filepath=args.output_filepath)