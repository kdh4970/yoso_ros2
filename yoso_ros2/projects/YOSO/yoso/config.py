from detectron2.config import CfgNode as CN

def add_yoso_config(cfg):
    cfg.MODEL.YOSO = CN()
    cfg.MODEL.YOSO.SIZE_DIVISIBILITY = 32
    cfg.MODEL.YOSO.NUM_CLASSES = 133
    cfg.MODEL.YOSO.NUM_STAGES = 2
    
    cfg.MODEL.YOSO.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.YOSO.HIDDEN_DIM = 256
    cfg.MODEL.YOSO.AGG_DIM = 128
    cfg.MODEL.YOSO.NUM_PROPOSALS = 100
    cfg.MODEL.YOSO.CONV_KERNEL_SIZE_2D = 1
    cfg.MODEL.YOSO.CONV_KERNEL_SIZE_1D = 3
    cfg.MODEL.YOSO.NUM_CLS_FCS = 1
    cfg.MODEL.YOSO.NUM_MASK_FCS = 1

    cfg.MODEL.YOSO.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.YOSO.CLASS_WEIGHT = 2.0
    cfg.MODEL.YOSO.MASK_WEIGHT = 5.0
    cfg.MODEL.YOSO.DICE_WEIGHT = 5.0
    cfg.MODEL.YOSO.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.YOSO.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.YOSO.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.YOSO.TEMPERATIRE = 0.1

    cfg.MODEL.YOSO.TEST = CN()
    cfg.MODEL.YOSO.TEST.SEMANTIC_ON = False
    cfg.MODEL.YOSO.TEST.INSTANCE_ON = False
    cfg.MODEL.YOSO.TEST.PANOPTIC_ON = False
    cfg.MODEL.YOSO.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.YOSO.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.YOSO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.WEIGHT_DECAY_BIAS = None
    
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0

    cfg.INPUT.DATASET_MAPPER_NAME = "yoso_panoptic_lsj"
    cfg.INPUT.SIZE_DIVISIBILITY = -1
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0

    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
