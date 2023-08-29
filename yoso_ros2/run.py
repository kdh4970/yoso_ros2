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
import torch
import numba as nb

torch.multiprocessing.set_start_method('spawn')

# constants
image_scale = 5
# Image_height = 1536 * (image_scale/10)  #1536
# Image_width = 2048 * (image_scale/10)#2048
Image_height = 72 * image_scale  #720
Image_width = 128 * image_scale#1280

DEBUG = False
VISUALIZE = False
FAKE_TOPIC=False
class YosoNode(Node):

    def __init__(self):
        callback_group0 = MutuallyExclusiveCallbackGroup()
        callback_group1 = MutuallyExclusiveCallbackGroup()
        callback_group2 = MutuallyExclusiveCallbackGroup()
        super().__init__("yoso_node")
        self.image_topic0 = "/azure_kinect/master/rgb/image_raw"
        self.image_topic1 = "/azure_kinect/sub1/rgb/image_raw"
        self.image_topic2 = "/azure_kinect/sub2/rgb/image_raw"
        self.image_sub0 = self.create_subscription(Image, self.image_topic0, self.image_callback0, 2, callback_group=callback_group0)
        self.image_sub1 = self.create_subscription(Image, self.image_topic1, self.image_callback1, 2, callback_group=callback_group1)
        self.image_sub2 = self.create_subscription(Image, self.image_topic2, self.image_callback2, 2, callback_group=callback_group2)
        self.seg_pub0 = self.create_publisher(Image, "/yoso_node/master", 1)
        self.seg_pub1 = self.create_publisher(Image, "/yoso_node/sub1", 1)
        self.seg_pub2 = self.create_publisher(Image, "/yoso_node/sub2", 1)
        self.input_image0 = None
        self.input_image1 = None
        self.input_image2 = None
        self.result_frame0 = None
        self.result_frame1 = None
        self.result_frame2 = None
        self.seg_msg0 = None
        self.seg_msg1 = None
        self.seg_msg2 = None
        self.bridge = CvBridge()
        self.demo = None
        self.period=0.05
        if FAKE_TOPIC: self.timer = self.create_timer(self.period, self.timer_callback)
        self.hz = 1/self.period

    def timer_callback(self):
        nanosecMax = 1000000000
        if self.seg_msg0 is not None:
            if self.seg_msg0.header.stamp.nanosec + int(nanosecMax/self.hz) >= nanosecMax:
                self.seg_msg0.header.stamp.sec += 1
                self.seg_msg0.header.stamp.nanosec -= nanosecMax - int(nanosecMax/self.hz)
            else:
                self.seg_msg0.header.stamp.nanosec += int(nanosecMax/self.hz)
            self.seg_pub0.publish(self.seg_msg0)
        if self.seg_msg1 is not None:
            if self.seg_msg1.header.stamp.nanosec + int(nanosecMax/self.hz) >= nanosecMax:
                self.seg_msg1.header.stamp.sec += 1
                self.seg_msg1.header.stamp.nanosec -= nanosecMax - int(nanosecMax/self.hz)
            else:
                self.seg_msg1.header.stamp.nanosec += int(nanosecMax/self.hz)
            self.seg_pub1.publish(self.seg_msg1)
        if self.seg_msg2 is not None:
            if self.seg_msg2.header.stamp.nanosec + int(nanosecMax/self.hz) >= nanosecMax:
                self.seg_msg2.header.stamp.sec += 1
                self.seg_msg2.header.stamp.nanosec -= nanosecMax - int(nanosecMax/self.hz)
            else:
                self.seg_msg2.header.stamp.nanosec += int(nanosecMax/self.hz)
            self.seg_pub2.publish(self.seg_msg2)
        # print("|| time stamp orig : ", self.seg_msg0.header.stamp.nanosec)
        # print("|| time stamp add  : ", self.seg_msg0.header.stamp.nanosec + int(nanosecMax/100))
        


    def getcfg(self,cfg):
        self.demo = VisualizationDemo(cfg, parallel=False)
        self.demo1 = VisualizationDemo(cfg, parallel=False)
        self.demo2 = VisualizationDemo(cfg, parallel=False)

    def change_scale(self,pos):
        if pos==0: return
        global image_scale, Image_height, Image_width
        image_scale = pos
        # Image_height = 1536 * (image_scale/10)
        # Image_width = 2048 * (image_scale/10)
        Image_height = 72 * image_scale
        Image_width = 128 * image_scale

    def image_callback0(self, msg):
        try :
            start_time = timeit.default_timer()
            self.input_image0 = cv2.resize(self.bridge.imgmsg_to_cv2(msg, "bgr8"), (int(Image_width),int(Image_height)))# 1280 720
            if DEBUG:
                resize_time = timeit.default_timer()
                print("+++++image callback0"); #print("running time : ", 1000 * (end_time-start_time)); print("FPS : ", 1/(end_time-start_time))
                print("resize time           : ",1000*(resize_time-start_time))
            # self.result_frame0, self.result_panoptic_seg0, self.segments_info0 = self.demo.run_on_azure(self.input_image0)
            self.result_panoptic_seg0, self.segments_info0 = self.demo.run_on_azure(self.input_image0)
            if DEBUG:
                seg_time = timeit.default_timer()
                print("prediction time       : ", 1000 * (seg_time-resize_time))
            self.gen_seg_mask(0, msg.header, self.result_panoptic_seg0, self.segments_info0)
            # self.debug_panoptic()

            # end_time = timeit.default_timer()
            # print("+++ callback time : ", 1000 * (end_time-start_time))
            # FPS = 1./(end_time - start_time)
            # if VISUALIZE:
            #     cv2.getTrackbarPos('Scale','cam 0')
            #     cv2.putText(self.result_frame0, str(Image_width) + "x" + str(Image_height), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #     cv2.putText(self.result_frame0, "FPS : {:.1f}".format(FPS), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #     cv2.imshow("cam 0", self.result_frame0)
            #     cv2.waitKey(1)
            total_time = timeit.default_timer()
            print(f"callback0 processed : {1000 * (total_time-start_time)} ms")
        except CvBridgeError as e:
            print(e)
    def image_callback1(self, msg):
        try :
            start_time = timeit.default_timer()
            self.input_image1 = cv2.resize(self.bridge.imgmsg_to_cv2(msg, "bgr8"), (int(Image_width),int(Image_height)))# 1280 720
            if DEBUG:
                resize_time = timeit.default_timer()
                print("+++++image callback1"); #print("running time : ", 1000 * (end_time-start_time)); print("FPS : ", 1/(end_time-start_time))
                print("resize time           : ",1000*(resize_time-start_time))
            # self.result_frame1, self.result_panoptic_seg1, self.segments_info1 = self.demo1.run_on_azure(self.input_image1)
            self.result_panoptic_seg1, self.segments_info1 = self.demo1.run_on_azure(self.input_image1)
            if DEBUG: 
                seg_time = timeit.default_timer()
                print("prediction time       : ", 1000 * (seg_time-resize_time))
            self.gen_seg_mask(1, msg.header, self.result_panoptic_seg1, self.segments_info1)

            # end_time = timeit.default_timer()
            # FPS = 1./(end_time - start_time)
            
            # if VISUALIZE:
            #     cv2.putText(self.result_frame1, "FPS : {:.1f}".format(FPS), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #     cv2.imshow("cam 1", self.result_frame1)
            #     cv2.waitKey(1)
            total_time = timeit.default_timer()
            print(f"callback1 processed : {1000 * (total_time-start_time)} ms")
        except CvBridgeError as e:
            print(e)
    def image_callback2(self, msg):
        try :
            start_time = timeit.default_timer()
            self.input_image2 = cv2.resize(self.bridge.imgmsg_to_cv2(msg, "bgr8"), (int(Image_width),int(Image_height)))# 1280 720
            if DEBUG: 
                resize_time = timeit.default_timer()
                print("+++++image callback2"); #rint("running time : ", 1000 * (end_time-start_time)); print("FPS : ", 1/(end_time-start_time))
                print("resize time           : ",1000*(resize_time-start_time))
            # self.result_frame2, self.result_panoptic_seg2, self.segments_info2 = self.demo2.run_on_azure(self.input_image2)
            self.result_panoptic_seg2, self.segments_info2 = self.demo2.run_on_azure(self.input_image2)
            if DEBUG: 
                seg_time = timeit.default_timer()
                print("prediction time       : ", 1000 * (seg_time-resize_time))
            self.gen_seg_mask(2, msg.header, self.result_panoptic_seg2, self.segments_info2)

            # end_time = timeit.default_timer()
            # FPS = 1./(end_time - start_time)
            
            # if VISUALIZE:
            #     cv2.putText(self.result_frame2, "FPS : {:.1f}".format(FPS), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #     cv2.imshow("cam 2", self.result_frame2)
            #     cv2.waitKey(1)
            total_time = timeit.default_timer()
            print(f"callback2 processed : {1000 * (total_time-start_time)} ms")
        except CvBridgeError as e:
            print(e)

    def gen_seg_mask(self,camnum,header,panoptic_seg,seg_info):
        if DEBUG: start_time = timeit.default_timer()
        # convert tensor to numpy array
        temp = panoptic_seg.cpu().numpy()
        if DEBUG: checkpoint = timeit.default_timer()
        """
        ### generate panoptic mask
        # t0=timeit.default_timer()
        # panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
        # for i in range(temp.shape[0]):
        #     for j in range(temp.shape[1]):
        #         if temp[i,j] == 0: panoptic_mask[i,j] = 0
        #         else: panoptic_mask[i,j] = seg_info[temp[i,j]-1]['category_id']
        # t1=timeit.default_timer()
        # print("[mask gen] doubled for loop : ", 1000 * (t1-t0))

        # #############################################
        # chkpoint0 = timeit.default_timer()
        # def get_category_id(idx):
        #     return 1 + seg_info[idx-1]['category_id']
        
        # panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
        # vfunc = np.vectorize(get_category_id)
        # nonzero = np.nonzero(temp)
        # panoptic_mask[nonzero] = vfunc(temp[nonzero])
        # chkpoint1 = timeit.default_timer()
        # print("[mask gen] vectorize nonzero: ", 1000 * (chkpoint1-chkpoint0))
        # #############################################
        # chkpoint2 = timeit.default_timer()
        # panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
        # lambda_func = lambda idx: 1 + seg_info[idx-1]['category_id'] if idx != 0 else 0
        # vfunc = np.vectorize(lambda_func)
        # nonzero = np.nonzero(temp)
        # panoptic_mask[nonzero] = vfunc(temp[nonzero])
        # chkpoint3 = timeit.default_timer() 
        # print("[mask gen] vectorize lambda : ", 1000 * (chkpoint3-chkpoint2))
        # #############################################
        # chkpoint2 = timeit.default_timer()
        # panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
        # lambda_func = lambda idx: 1 + seg_info[idx-1]['category_id'] if idx != 0 else 0
        # # gen panoptic mask using map
        # panoptic_mask = np.array(list(map(lambda_func,temp.flatten()))).reshape(temp.shape)
        # chkpoint3 = timeit.default_timer() 
        # print("[mask gen] lambda map       : ", 1000 * (chkpoint3-chkpoint2))
        # ##############################################
        # chkpoint0 = timeit.default_timer()
        # def get_category_id(idx):
        #     if idx==0: return 0
        #     else: return 1 + seg_info[idx-1]['category_id']
        
        # panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
        # vfunc = np.vectorize(get_category_id)
        # # gen panoptic mask using vfunc
        # panoptic_mask = vfunc(temp)
        # chkpoint1 = timeit.default_timer()
        # print("[mask gen] vectorize         : ", 1000 * (chkpoint1-chkpoint0))
        ##############################################"""
        
        panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
        for i in range(len(seg_info)): panoptic_mask[temp==i+1] = seg_info[i]['category_id']+1
            # len seg_info=10; i = 0~9 >>i+1=1~10  
            # seg_info[0] == id:1 /
            # below code is numpy array support 
            # temp = 7 7 7      temp==7 >>> 1 1 1       
            #        7 5 7                  1 0 1
            #        5 5 5                  0 0 0
            # panoptic_mask[temp==7] = 10
            # >>> panoptic_mask = 10 10 10
            #                     10 0  10
            #                     0  0  0 
        
        ############################################3#
        # chk0=timeit.default_timer()
        # panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
        # for i in range(len(seg_info)):# len seg_info=10; i = 0~9 >>i+1=1~10  
            # panoptic_mask = np.where(temp==i+1,seg_info[i]['category_id']+1,panoptic_mask)
        #     # panoptic_mask[temp==i+1] = seg_info[i]['category_id']+1
        # chk1=timeit.default_timer()
        # print("[mask gen] dict loop np.where: ", 1000 * (chk1-chk0))
                
        
        # show mask image
        if VISUALIZE:
            # print(panoptic_mask)
            # print(panoptic_mask.shape)
            windowname="Panoptic Mask "+str(camnum)
            cv2.imshow(windowname, panoptic_mask)
            # cv2.waitKey(1)

        if DEBUG: checkpoint2 = timeit.default_timer()
        Seg_msg = self.bridge.cv2_to_imgmsg(panoptic_mask, "mono8")
        Seg_msg.header = header
        
        if DEBUG: checkpoint3 = timeit.default_timer()
        
        if camnum==0: 
            self.seg_pub0.publish(Seg_msg)
            self.seg_msg0 = Seg_msg
        elif camnum==1: 
            self.seg_pub1.publish(Seg_msg)
            self.seg_msg1 = Seg_msg
        elif camnum==2: 
            self.seg_pub2.publish(Seg_msg)
            self.seg_msg2 = Seg_msg
        if DEBUG:
            end_time = timeit.default_timer()
            print("tensor to numpy time  : ", 1000 * (checkpoint-start_time))
            print("mask gen time         : ", 1000 * (checkpoint2-checkpoint))
            print("mask to msg copy time : ", 1000 * (checkpoint3-checkpoint2))
            print("msg pub time          : ", 1000 * (end_time-checkpoint3))

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


def main(args=sys.argv[1:]):
    rclpy.init()
    node=YosoNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    print("ROS2 Node started!")
    # rclpy.spin(node)
    
    # mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    node.getcfg(cfg)
    
    

    # demo = VisualizationDemo(cfg, parallel=False)
    if VISUALIZE:
        cv2.namedWindow('cam 0', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Panoptic Mask 0', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Panoptic Mask 1', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Panoptic Mask 2', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Scale', 'cam 0', image_scale, 10, node.change_scale)
    
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