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
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from demo.predictor import VisualizationDemo
from demo.config import add_yoso_config
from projects.YOSO.yoso.segmentator import YOSO
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from message_filters import ApproximateTimeSynchronizer, Subscriber
import torch

torch.multiprocessing.set_start_method('spawn',force=True)

# usgin torch cuda gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


logger = setup_logger(name="YOSO_ROS2")
log = logger.info
# constants
image_scale = 5
# Image_height = 1536 * (image_scale/10)  #1536
# Image_width = 2048 * (image_scale/10)#2048
Image_height = 72 * image_scale  #720
Image_width = 128 * image_scale#1280

DEBUG = False
VISUALIZE = False
FAKE_TOPIC = False
# class YosoNode(Node):
#     ## 싱크 안맞췄음.
#     def __init__(self):
#         callback_group0 = MutuallyExclusiveCallbackGroup()
#         callback_group1 = MutuallyExclusiveCallbackGroup()
#         callback_group2 = MutuallyExclusiveCallbackGroup()
#         super().__init__("yoso_node")
#         self.image_topic0 = "/azure_kinect/master/rgb/image_raw"
#         self.image_topic1 = "/azure_kinect/sub1/rgb/image_raw"
#         self.image_topic2 = "/azure_kinect/sub2/rgb/image_raw"
#         self.image_sub0 = self.create_subscription(Image, self.image_topic0, self.image_callback0, 2, callback_group=callback_group0)
#         self.image_sub1 = self.create_subscription(Image, self.image_topic1, self.image_callback1, 2, callback_group=callback_group1)
#         self.image_sub2 = self.create_subscription(Image, self.image_topic2, self.image_callback2, 2, callback_group=callback_group2)
#         self.seg_pub0 = self.create_publisher(Image, "/yoso_node/master", 1)
#         self.seg_pub1 = self.create_publisher(Image, "/yoso_node/sub1", 1)
#         self.seg_pub2 = self.create_publisher(Image, "/yoso_node/sub2", 1)
#         self.input_image0 = None
#         self.input_image1 = None
#         self.input_image2 = None
#         self.result_frame0 = None
#         self.result_frame1 = None
#         self.result_frame2 = None
#         self.seg_msg0 = None
#         self.seg_msg1 = None
#         self.seg_msg2 = None
#         self.bridge = CvBridge()
#         self.demo = None
#         self.period=0.05
#         if FAKE_TOPIC: self.timer = self.create_timer(self.period, self.timer_callback)
#         self.hz = 1/self.period
        

#     def timer_callback(self):
#         nanosecMax = 1000000000
#         if self.seg_msg0 is not None:
#             if self.seg_msg0.header.stamp.nanosec + int(nanosecMax/self.hz) >= nanosecMax:
#                 self.seg_msg0.header.stamp.sec += 1
#                 self.seg_msg0.header.stamp.nanosec -= nanosecMax - int(nanosecMax/self.hz)
#             else:
#                 self.seg_msg0.header.stamp.nanosec += int(nanosecMax/self.hz)
#             self.seg_pub0.publish(self.seg_msg0)
#         if self.seg_msg1 is not None:
#             if self.seg_msg1.header.stamp.nanosec + int(nanosecMax/self.hz) >= nanosecMax:
#                 self.seg_msg1.header.stamp.sec += 1
#                 self.seg_msg1.header.stamp.nanosec -= nanosecMax - int(nanosecMax/self.hz)
#             else:
#                 self.seg_msg1.header.stamp.nanosec += int(nanosecMax/self.hz)
#             self.seg_pub1.publish(self.seg_msg1)
#         if self.seg_msg2 is not None:
#             if self.seg_msg2.header.stamp.nanosec + int(nanosecMax/self.hz) >= nanosecMax:
#                 self.seg_msg2.header.stamp.sec += 1
#                 self.seg_msg2.header.stamp.nanosec -= nanosecMax - int(nanosecMax/self.hz)
#             else:
#                 self.seg_msg2.header.stamp.nanosec += int(nanosecMax/self.hz)
#             self.seg_pub2.publish(self.seg_msg2)
#         # print("|| time stamp orig : ", self.seg_msg0.header.stamp.nanosec)
#         # print("|| time stamp add  : ", self.seg_msg0.header.stamp.nanosec + int(nanosecMax/100))
        


#     def getcfg(self,cfg):
#         self.demo = VisualizationDemo(cfg, parallel=False)
#         self.demo1 = VisualizationDemo(cfg, parallel=False)
#         self.demo2 = VisualizationDemo(cfg, parallel=False)

#     def change_scale(self,pos):
#         if pos==0: return
#         global image_scale, Image_height, Image_width
#         image_scale = pos
#         # Image_height = 1536 * (image_scale/10)
#         # Image_width = 2048 * (image_scale/10)
#         Image_height = 72 * image_scale
#         Image_width = 128 * image_scale

#     def image_callback0(self, msg):
#         try :
#             start_time = timeit.default_timer()
#             self.input_image0 = cv2.resize(self.bridge.imgmsg_to_cv2(msg, "bgr8"), (int(Image_width),int(Image_height)))# 1280 720
#             if DEBUG:
#                 resize_time = timeit.default_timer()
#                 print("+++++image callback0"); #print("running time : ", 1000 * (end_time-start_time)); print("FPS : ", 1/(end_time-start_time))
#                 print("resize time           : ",1000*(resize_time-start_time))
#             # self.result_frame0, self.result_panoptic_seg0, self.segments_info0 = self.demo.run_on_azure(self.input_image0)
#             self.result_panoptic_seg0, self.segments_info0 = self.demo.run_on_azure(self.input_image0)
#             if DEBUG:
#                 seg_time = timeit.default_timer()
#                 print("prediction time       : ", 1000 * (seg_time-resize_time))
#             self.gen_seg_mask(0, msg.header, self.result_panoptic_seg0, self.segments_info0)
#             # self.debug_panoptic()

#             # end_time = timeit.default_timer()
#             # print("+++ callback time : ", 1000 * (end_time-start_time))
#             # FPS = 1./(end_time - start_time)
#             # if VISUALIZE:
#             #     cv2.getTrackbarPos('Scale','cam 0')
#             #     cv2.putText(self.result_frame0, str(Image_width) + "x" + str(Image_height), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             #     cv2.putText(self.result_frame0, "FPS : {:.1f}".format(FPS), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             #     cv2.imshow("cam 0", self.result_frame0)
#             #     cv2.waitKey(1)
#             total_time = timeit.default_timer()
#             print(f"Processing time of Cam 0 : {1000 * (total_time-start_time):.0f} ms          FPS : {1/(total_time-start_time):.1f}",end="\r")
                
#         except CvBridgeError as e:
#             print(e)
#     def image_callback1(self, msg):
#         try :
#             start_time = timeit.default_timer()
#             self.input_image1 = cv2.resize(self.bridge.imgmsg_to_cv2(msg, "bgr8"), (int(Image_width),int(Image_height)))# 1280 720
#             if DEBUG:
#                 resize_time = timeit.default_timer()
#                 print("+++++image callback1"); #print("running time : ", 1000 * (end_time-start_time)); print("FPS : ", 1/(end_time-start_time))
#                 print("resize time           : ",1000*(resize_time-start_time))
#             # self.result_frame1, self.result_panoptic_seg1, self.segments_info1 = self.demo1.run_on_azure(self.input_image1)
#             self.result_panoptic_seg1, self.segments_info1 = self.demo1.run_on_azure(self.input_image1)
#             if DEBUG: 
#                 seg_time = timeit.default_timer()
#                 print("prediction time       : ", 1000 * (seg_time-resize_time))
#             self.gen_seg_mask(1, msg.header, self.result_panoptic_seg1, self.segments_info1)

#             # end_time = timeit.default_timer()
#             # FPS = 1./(end_time - start_time)
            
#             # if VISUALIZE:
#             #     cv2.putText(self.result_frame1, "FPS : {:.1f}".format(FPS), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             #     cv2.imshow("cam 1", self.result_frame1)
#             #     cv2.waitKey(1)
#             total_time = timeit.default_timer()
#             # print(f"Cam 1 : {1000 * (total_time-start_time)} ms",end="\r")
#             print(f"Processing time of Cam 1 : {1000 * (total_time-start_time):.0f} ms          FPS : {1/(total_time-start_time):.1f}",end="\r")
            
#         except CvBridgeError as e:
#             print(e)
#     def image_callback2(self, msg):
#         try :
#             start_time = timeit.default_timer()
#             self.input_image2 = cv2.resize(self.bridge.imgmsg_to_cv2(msg, "bgr8"), (int(Image_width),int(Image_height)))# 1280 720
#             if DEBUG: 
#                 resize_time = timeit.default_timer()
#                 print("+++++image callback2"); #rint("running time : ", 1000 * (end_time-start_time)); print("FPS : ", 1/(end_time-start_time))
#                 print("resize time           : ",1000*(resize_time-start_time))
#             # self.result_frame2, self.result_panoptic_seg2, self.segments_info2 = self.demo2.run_on_azure(self.input_image2)
#             self.result_panoptic_seg2, self.segments_info2 = self.demo2.run_on_azure(self.input_image2)
#             if DEBUG: 
#                 seg_time = timeit.default_timer()
#                 print("prediction time       : ", 1000 * (seg_time-resize_time))
#             self.gen_seg_mask(2, msg.header, self.result_panoptic_seg2, self.segments_info2)

#             # end_time = timeit.default_timer()
#             # FPS = 1./(end_time - start_time)
            
#             # if VISUALIZE:
#             #     cv2.putText(self.result_frame2, "FPS : {:.1f}".format(FPS), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             #     cv2.imshow("cam 2", self.result_frame2)
#             #     cv2.waitKey(1)
#             total_time = timeit.default_timer()
#             print(f"Processing time of Cam 2 : {1000 * (total_time-start_time):.0f} ms          FPS : {1/(total_time-start_time):.1f}",end="\r")
            
#         except CvBridgeError as e:
#             print(e)

#     def gen_seg_mask(self,camnum,header,panoptic_seg,seg_info):
#         if DEBUG: start_time = timeit.default_timer()
#         # convert tensor to numpy array
#         temp = panoptic_seg.cpu().numpy()
#         if DEBUG: checkpoint = timeit.default_timer()
#         panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
#         for i in range(len(seg_info)): panoptic_mask[temp==i+1] = seg_info[i]['category_id']+1
        
#         if VISUALIZE:
#             # print(panoptic_mask)
#             # print(panoptic_mask.shape)
#             windowname="Panoptic Mask "+str(camnum)
#             cv2.imshow(windowname, panoptic_mask)
#             # cv2.waitKey(1)

#         if DEBUG: checkpoint2 = timeit.default_timer()
#         Seg_msg = self.bridge.cv2_to_imgmsg(panoptic_mask, "mono8")
#         Seg_msg.header = header
        
#         if DEBUG: checkpoint3 = timeit.default_timer()
        
#         if camnum==0: 
#             self.seg_pub0.publish(Seg_msg)
#             self.seg_msg0 = Seg_msg
#         elif camnum==1: 
#             self.seg_pub1.publish(Seg_msg)
#             self.seg_msg1 = Seg_msg
#         elif camnum==2: 
#             self.seg_pub2.publish(Seg_msg)
#             self.seg_msg2 = Seg_msg
#         if DEBUG:
#             end_time = timeit.default_timer()
#             print("tensor to numpy time  : ", 1000 * (checkpoint-start_time))
#             print("mask gen time         : ", 1000 * (checkpoint2-checkpoint))
#             print("mask to msg copy time : ", 1000 * (checkpoint3-checkpoint2))
#             print("msg pub time          : ", 1000 * (end_time-checkpoint3))

#     def debug_panoptic(self):
#         temp = self.result_panoptic_seg0.tolist() # convert tensor to list
#         print("!!START===============================")
#         # print(temp[0][0]) # pixel's seg id
#         print(temp[350]); print(len(temp[0]))
#         print(self.segments_info0)
#         print(self.segments_info0[0]['id'])
#         print(len(self.segments_info0))
#         # print(self.result_panoptic_seg0.size())
#         print("!!END===============================")



# class YosoNode_dev(Node):
#     def __init__(self,args,input_camera_name:str) -> None:
#         rclpy.init()

#         super().__init__("YOSO_" + input_camera_name)

#         # Define Topic and Subscriber and Publisher
#         self._target_cam = input_camera_name
#         self._rgb_topic_name = "/azure_kinect/" + input_camera_name + "/rgb/image_raw"
#         self._rgb_image_subscriber = self.create_subscription(Image, self._rgb_topic_name, self.single_callback, 2)
#         self._seg_mask_pub = self.create_publisher(Image, "/yoso_node/" + input_camera_name , 1)

#         self.bridge = CvBridge()

#         args = get_parser().parse_args()
#         setup_logger(name="fvcore")
#         logger = setup_logger()
#         logger.info("Arguments: " + str(args))

#         cfg = setup_cfg(args)
#         self.demo = VisualizationDemo(cfg, parallel=False)
#         rclpy.spin(self)
#         rclpy.shutdown()

#     def change_scale(self,pos)-> None:
#         if pos==0: return
#         global image_scale, Image_height, Image_width
#         image_scale = pos
#         # Image_height = 1536 * (image_scale/10)
#         # Image_width = 2048 * (image_scale/10)
#         Image_height = 72 * image_scale
#         Image_width = 128 * image_scale

#     def run(self,args) -> None:
#         args = get_parser().parse_args()
#         setup_logger(name="fvcore")
#         logger = setup_logger()
#         logger.info("Arguments: " + str(args))

#         cfg = setup_cfg(args)
#         self.demo = VisualizationDemo(cfg, parallel=False)
#         rclpy.spin(self)

#     def generate_single_segmask(self,header,panoptic_seg,seg_info):
#         if DEBUG: start_time = timeit.default_timer()
#         # convert tensor to numpy array
#         temp = panoptic_seg.cpu().numpy()
#         if DEBUG: checkpoint = timeit.default_timer()
#         panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
#         for i in range(len(seg_info)): panoptic_mask[temp==i+1] = seg_info[i]['category_id']+1

#         if DEBUG: checkpoint2 = timeit.default_timer()
#         Seg_msg = self.bridge.cv2_to_imgmsg(panoptic_mask, "mono8")
#         Seg_msg.header = header
        
#         if DEBUG: checkpoint3 = timeit.default_timer()
        
        
#         self._seg_mask_pub.publish(Seg_msg)
#         if DEBUG:
#             end_time = timeit.default_timer()
#             print("tensor to numpy time  : ", 1000 * (checkpoint-start_time))
#             print("mask gen time         : ", 1000 * (checkpoint2-checkpoint))
#             print("mask to msg copy time : ", 1000 * (checkpoint3-checkpoint2))
#             print("msg pub time          : ", 1000 * (end_time-checkpoint3))

#     def single_callback(self, msg)-> None:
#         try :
#             start_time = timeit.default_timer()
#             input_image = cv2.resize(self.bridge.imgmsg_to_cv2(msg, "bgr8"), (int(Image_width),int(Image_height)))# 1280 720
#             if DEBUG:
#                 resize_time = timeit.default_timer()
#                 print("resize time           : ",1000*(resize_time-start_time))
#             result_panoptic_seg, segments_info = self.demo.run_on_azure(input_image)
#             if DEBUG:
#                 seg_time = timeit.default_timer()
#                 print("prediction time       : ", 1000 * (seg_time-resize_time))
#             self.generate_single_segmask(msg.header, result_panoptic_seg, segments_info)

#             total_time = timeit.default_timer()
#             print(f"Processing time of {self._target_cam} : {1000 * (total_time-start_time):.0f} ms          FPS : {1/(total_time-start_time):.1f}",end="\n")
                
#         except CvBridgeError as e:
#             print(e)

class SubProcess():
    def __init__(self,cfg):
        self.demo = VisualizationDemo(cfg, parallel=False)

    def generate_and_publish_segmask(self, image, index) -> None:
        log(f"Job {index} >>> Start")
        panoptic_seg, seg_info = self.demo.run_on_azure(image)
        temp = panoptic_seg.cpu().numpy()
        panoptic_mask = np.zeros(temp.shape, dtype=np.uint8)
        for i in range(len(seg_info)): panoptic_mask[temp==i+1] = seg_info[i]['category_id']+1

        log(f"Job {index} <<< Done")
        return panoptic_mask

# submodule = []

# def init_worker(cfg):
#     global submodule
#     submodule.append(SubProcess(cfg))

submodule = {}

def worker(args):
    cfg, image, index = args
    pid = current_process().pid
    if pid not in submodule:
        log(f"Process {pid} >>> Creating Model on Device...")
        submodule[pid] = SubProcess(cfg)
    return submodule[pid].generate_and_publish_segmask(image,index)



class YosoNode_240126(Node):
    def __init__(self,cfg):
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

        log("Creating Process Pool...")
        self.process_pool = mp.Pool(processes=3)

        
    def __exit__(self, exc_type, exc_value, traceback):
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
            input_image.append(cv2.resize(self.bridge.imgmsg_to_cv2(msg, "bgr8"), (int(Image_width),int(Image_height))))# 1280 720
            header.append(msg.header)
        
        # allocate process for call goto_model
        log("Allocating Task to Process Pool...")
        indices = [0,1,2]
        # breakpoint()
        self.mask = self.process_pool.map(worker, zip(self.cfg, input_image, indices),chunksize=1)
        # publish
        log("Publishing Segmentation Mask...")
        for i in indices:
            self.publish_segmask(self.mask[i], header[i], i)
        end = timeit.default_timer()
        log(f"Total Processing Time : {1000 * (end-start):.0f} ms          FPS : {1/(end-start):.1f}")
        self.count += 1

    def publish_segmask(self, mask, header, index) -> None:
        Seg_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
        Seg_msg.header = header
        self.seg_pub[index].publish(Seg_msg)


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

def ctrl_c_handler(sig, frame):
    print('Good bye!')
    sys.exit(0)

def main(args=sys.argv[1:]):
    args = get_parser().parse_args()
    
    log("YOSO Arguments: " + str(args))
    cfg = setup_cfg(args)

    rclpy.init()
    # node = YosoNode_dev("master")
    log("Staring YOSO ROS2 Node...")
    node = YosoNode_240126(cfg)
    

    rclpy.spin(node)

    # # create 3 process for 3 cameras by using mp
    # # node0 = YosoNode_dev("master")
    # # node1 = YosoNode_dev("sub1")
    # # node2 = YosoNode_dev("sub2")
    # # node0.run(args)
    # p0 = mp.Process(target=YosoNode_dev,args=(args,"master",))
    # p1 = mp.Process(target=YosoNode_dev,args=(args,"sub1",))
    # p2 = mp.Process(target=YosoNode_dev,args=(args,"sub2",))

    # # start 3 process
    # p0.start()
    # p1.start()
    # p2.start()

    # # # wait for 3 process
    # p0.join()
    # p1.join()
    # p2.join()




    # cfg = setup_cfg(args)
    # node.getcfg(cfg)
    
    # # clear the console terminal
    # os.system('clear')
    # print("===================== YOSO ROS2 Node =====================")
    # # demo = VisualizationDemo(cfg, parallel=False)
    # if VISUALIZE:
    #     cv2.namedWindow('cam 0', cv2.WINDOW_NORMAL)
    #     cv2.namedWindow('Panoptic Mask 0', cv2.WINDOW_NORMAL)
    #     cv2.namedWindow('Panoptic Mask 1', cv2.WINDOW_NORMAL)
    #     cv2.namedWindow('Panoptic Mask 2', cv2.WINDOW_NORMAL)
    #     cv2.createTrackbar('Scale', 'cam 0', image_scale, 10, node.change_scale)
    # if args.azure:
    #     try:
    #         rclpy.spin(node)
    #     finally:
    #         node.destroy_node()
    #         rclpy.shutdown()



    # nodecb0.destroy_node()
    # node1.destroy_node()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()