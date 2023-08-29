# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import MultiThreadedExecutor
import sys
import timeit
from rclpy.node import Node
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from demo.predictor import VisualizationDemo
from demo.config import add_yoso_config
from projects.YOSO.yoso.segmentator import YOSO
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup


# constants
image_scale = 5
Image_height = 72 * image_scale  #720
Image_width = 128 * image_scale#1280


CALLBACK_DEBUG = True
VISUALIZE = False
class YosoNode(Node):

    def __init__(self, camnum,window_name_cam,window_name_mask):
        # callback_group0 = MutuallyExclusiveCallbackGroup()
        # callback_group1 = MutuallyExclusiveCallbackGroup()
        # callback_group2 = MutuallyExclusiveCallbackGroup()
        # callback_group = ReentrantCallbackGroup()
        node_name = "yoso_node"+str(camnum)
        self.window_name_cam = window_name_cam
        self.window_name_mask = window_name_mask
        super().__init__(node_name)
        self.camnum = camnum
        if self.camnum==0: 
            self.image_topic = "/azure_kinect/master/rgb/image_raw"
            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 2)
            self.seg_pub = self.create_publisher(Image, "/yoso_node/master", 1)
        elif self.camnum==1: 
            self.image_topic = "/azure_kinect/sub1/rgb/image_raw"
            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 2)
            self.seg_pub = self.create_publisher(Image, "/yoso_node/sub1", 1)
        elif self.camnum==2: 
            self.image_topic = "/azure_kinect/sub2/rgb/image_raw"
            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 2)
            self.seg_pub = self.create_publisher(Image, "/yoso_node/sub2", 1)
        self.input_image = None
        self.result_frame = None
        self.bridge = CvBridge()
        self.demo = None

    def getcfg(self,cfg):
        self.demo = VisualizationDemo(cfg, parallel=False)

    def change_scale(self,pos):
        if pos==0: return
        global image_scale, Image_height, Image_width
        image_scale = pos
        Image_height = 72 * image_scale 
        Image_width = 128 * image_scale

    def image_callback(self, msg):
        try :
            start_time = timeit.default_timer()
            
            self.input_image = cv2.resize(self.bridge.imgmsg_to_cv2(msg, "bgr8"), (int(Image_width),int(Image_height)))# 1280 720
            resize_time = timeit.default_timer()
            
            if CALLBACK_DEBUG: print("+++++image callback" + str(self.camnum)); #print("running time : ", 1000 * (end_time-start_time)); print("FPS : ", 1/(end_time-start_time))
            print("resize time             : ", 1000 * (resize_time-start_time))
            # self.result_frame, self.result_panoptic_seg, self.segments_info = self.demo.run_on_azure(self.input_image)
            self.result_panoptic_seg, self.segments_info = self.demo.run_on_azure(self.input_image)
            seg_time = timeit.default_timer()
            print("prediction time         : ", 1000 * (seg_time-resize_time))
            self.publish_seg_result(msg.header, self.result_panoptic_seg, self.segments_info)
            # self.debug_panoptic()

            pub_time = timeit.default_timer()
            # print("total publish time     : ", 1000 * (pub_time-seg_time))
            # print("+++ callback time : ", 1000 * (end_time-start_time))
            # FPS = 1./(pub_time - start_time)
            # if VISUALIZE:
            #     # cv2.getTrackbarPos('Scale','cam 0')
            #     cv2.putText(self.result_frame, str(Image_width) + "x" + str(Image_height), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #     cv2.putText(self.result_frame, "FPS : {:.1f}".format(FPS), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #     cv2.imshow(self.window_name_cam, self.result_frame)
            #     cv2.waitKey(1)
            total_time = timeit.default_timer()
            # print("visualize time          : ", 1000 * (total_time-pub_time))
            print("total time              : ", 1000 * (total_time-start_time))
        except CvBridgeError as e:
            print(e)
    

    def publish_seg_result(self,header,panoptic_seg,seg_info):
        start_time = timeit.default_timer()
        # convert tensor to numpy array
        temp = panoptic_seg.cpu().numpy()
        checkpoint = timeit.default_timer()

        ### generate panoptic mask
        ### older method : require 150~200ms
        # panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
        # nonzero = np.nonzero(temp)
        # panoptic_mask[nonzero] = 1 + np.array([seg_info[idx-1]['category_id'] for idx in temp[nonzero]])

        ### new method : require 10~20ms
        panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
        for i in range(len(seg_info)):
            panoptic_mask[temp == i+1] = 1 + seg_info[i]['category_id']


        # show mask image
        if VISUALIZE:
            # print(panoptic_mask)
            # print(panoptic_mask.shape)
            
            cv2.imshow(self.window_name_mask, panoptic_mask)
            cv2.waitKey(1)

        chcekpoint5 = timeit.default_timer()
        Seg_msg = self.bridge.cv2_to_imgmsg(panoptic_mask, "mono8")
        Seg_msg.header = header
        
        checkpoint6 = timeit.default_timer()
        
        self.seg_pub.publish(Seg_msg)
        end_time = timeit.default_timer()
        print(">> tensor to numpy time : ", 1000 * (checkpoint-start_time))
        print(">> mask gen time        : ", 1000 * (chcekpoint5-checkpoint))
        print(">> msg copy time        : ", 1000 * (checkpoint6-chcekpoint5))
        print(">> msg publish time     : ", 1000 * (end_time-checkpoint6))
        # print(">> total time            : ", 1000 * (end_time-start_time))

    def debug_panoptic(self):
        temp = self.result_panoptic_seg0.tolist() # convert tensor to list
        print("!!START===============================")
        # print(temp[0][0]) # pixel's seg id
        print(temp[350]); print(len(temp[0]))
        print(self.segments_info0)
        print(self.segments_info0[0]['id'])
        print(len(self.segments_info0))
        # print(self.result_panoptic_seg0.size())
        print("!!END===============================")


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


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
    parser.add_argument("--camnum", default=0, type=int, help="Camera number")
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


def main(args=sys.argv[1:]):
    print("=============================== PROGRAM START ===============================")
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    window_name_cam = "Cam" + str(args.camnum)
    window_name_mask = "Panoptic Mask" + str(args.camnum)
    print("[DEBUG] Reading args done. Target cam number : {}".format(args.camnum))

    rclpy.init()
    node=YosoNode(args.camnum,window_name_cam,window_name_mask)
    node.getcfg(cfg)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    print("[DEBUG] ROS2 Node started.")
    # rclpy.spin(node)
    
    # mp.set_start_method("spawn", force=True)
    
    
    

    # demo = VisualizationDemo(cfg, parallel=False)
    if VISUALIZE:
        cv2.namedWindow(window_name_cam, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_name_mask, cv2.WINDOW_NORMAL)
        # cv2.createTrackbar('Scale', 'cam 0', image_scale, 10, node.change_scale)
    
    if args.azure:
        try:
            # rclpy.spin(node)
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()






if __name__ == "__main__":
    main()