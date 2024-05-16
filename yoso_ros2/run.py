# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp
from multiprocessing import current_process
import numpy as np
import os
import signal
import cv2
import rclpy
import sys
import timeit
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from demo.predictor import VisualizationDemo
from demo.config import add_yoso_config
from projects.YOSO.yoso.segmentator import YOSO
from message_filters import ApproximateTimeSynchronizer, Subscriber
import torch

from functools import lru_cache

torch.multiprocessing.set_start_method('spawn',force=True)

# usgin torch cuda gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logger = setup_logger(name="YOSO_ROS2")
log = logger.info
# constants
image_scale = 5
# Image_height = 1536 * (image_scale/10)  #1536
# Image_width = 2048 * (image_scale/10)#2048
Image_height = int(72 * image_scale)  #720
Image_width = int(128 * image_scale)#1280



subprocess = {}

def ctrl_c_handler(sig, frame):
    print('Good bye!')
    sys.exit(0)

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_yoso_config(cfg)
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.YOSO.TEST.OVERLAP_THRESHOLD = args.overlap_threshold
    cfg.MODEL.YOSO.TEST.OBJECT_MASK_THRESHOLD = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--azure", action="store_true", help="Take inputs from azure kinect.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.98,
        help="overlap threshold",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    
    return parser

@lru_cache(maxsize=1)
def generateZeromask(shape):
    return np.full(shape, 0, dtype=np.uint8)

class SubProcessManager():
    def __init__(self,cfg) -> None:
        self.demo = VisualizationDemo(cfg, parallel=False)

    def generate_and_publish_segmask(self, image, index):
        log(f"Job {index} >>> Start")
        panoptic_seg, seg_info = self.demo.run_on_azure(image)
        temp = panoptic_seg.cpu().numpy()
        
        # panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
        panoptic_mask = generateZeromask(temp.shape)
        
        for i in range(len(seg_info)): 
            category_val = seg_info[i]["category_id"]+1
            panoptic_mask[temp==i+1] = category_val


        log(f"Job {index} <<< Done")
        return panoptic_mask

    @staticmethod
    def worker(args):
        cfg, image, index = args
        pid = current_process().pid
        if pid not in subprocess:
            log(f"Process {pid} >>> Creating Model on Device...")
            subprocess[pid] = SubProcessManager(cfg)
        return subprocess[pid].generate_and_publish_segmask(image,index)


class YosoNode(Node):
    def __init__(self,cfg) -> None:
        super().__init__("yoso_node")

        log("Establishing ROS2 Node Connection...")
        image_topic = [
            "/azure_kinect/master/rgb/image_raw",
            "/azure_kinect/sub1/rgb/image_raw",
            "/azure_kinect/sub2/rgb/image_raw"
        ]
        self.synced_sub = ApproximateTimeSynchronizer([
            Subscriber(self, Image, image_topic[0]),
            Subscriber(self, Image, image_topic[1]),
            Subscriber(self, Image, image_topic[2])
        ], 10, 0.1, allow_headerless=True)
        self.synced_sub.registerCallback(self.synced_callback)
        self.seg_pub = [
            self.create_publisher(Image, "/yoso_node/master", 1),
            self.create_publisher(Image, "/yoso_node/sub1", 1),
            self.create_publisher(Image, "/yoso_node/sub2", 1)
        ]
        self.bridge = CvBridge()
        self.mask = [None, None, None]
        self.cfg = [cfg] * 3
        self.count = 0
        self.indices = [0,1,2]

        log("Creating Process Pool...",)
        self.process_pool = mp.Pool(processes=3)

        
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.process_pool.close()
        self.process_pool.join()
        log("Process Pool Closed")
        super().__exit__(exc_type, exc_value, traceback)

    def synced_callback(self, *msgs) -> None:
        log(f"iteration ===========> {self.count}")
        start = timeit.default_timer()
        input_image = []
        header = []
        for msg in msgs:
            input_image.append(cv2.resize(self.bridge.imgmsg_to_cv2(msg, "bgr8"), (Image_width,Image_height)))# 1280 720
            header.append(msg.header)
        
        # allocate process for call goto_model
        
        # breakpoint()
        self.mask = self.process_pool.map(SubProcessManager.worker, zip(self.cfg, input_image, self.indices),chunksize=1)
        
        # publish
        for i in self.indices:
            self.publish_segmask(self.mask[i], header[i], i)
        end = timeit.default_timer()
        log(f"Total Processing Time : {1000 * (end-start):.0f} ms          FPS : {1/(end-start):.1f}")
        self.count += 1

    def publish_segmask(self, mask, header, index) -> None:
        Seg_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
        Seg_msg.header = header
        self.seg_pub[index].publish(Seg_msg)

def main(args=sys.argv[1:]):
    args = get_parser().parse_args()
    
    log("YOSO Arguments: " + str(args))
    cfg = setup_cfg(args)

    rclpy.init()
    # node = YosoNode_dev("master")
    log("Staring YOSO ROS2 Node...")
    node = YosoNode(cfg)
    

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()