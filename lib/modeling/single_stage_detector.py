import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from collections import namedtuple
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from core.config import cfg
from builder.anchor_builder import Anchors
from builder.target_assigner import TargetAssigner
from builder.layer_builder import LayerBuilder
from dataset.placeholders import PlaceHolders
from modeling.head_builder import HeadBuilder 
from builder.encoder_builder import EncoderDecoder
from builder.postprocessor import PostProcessor
from builder.loss_builder import LossBuilder
import utils.tf_util as tf_util
from utils.box_3d_utils import transfer_box3d_to_corners 
from utils.model_util import *
from img_vgg_pyramid import ImgVggPyr
import projection
from utils.layers_util import *

from utils.tf_ops.grouping.tf_grouping import *
from utils.tf_ops.sampling.tf_sampling import *
from utils.tf_ops.interpolation.tf_interpolate import *

import dataset.maps_dict as maps_dict

class SingleStageDetector:
    def __init__(self, batch_size, is_training):
        self.batch_size = batch_size
        self.is_training = is_training

        # placeholders
        self.placeholders_builder = PlaceHolders(self.batch_size) 
        self.placeholders_builder.get_placeholders()
        self.placeholders = self.placeholders_builder.placeholders

        self.cls_list = cfg.DATASET.KITTI.CLS_LIST
        self.cls2idx = dict([(cls, i + 1) for i, cls in enumerate(self.cls_list)])
        self.idx2cls = dict([(i + 1, cls) for i, cls in enumerate(self.cls_list)])

        # anchor_builder
        self.anchor_builder = Anchors(0, self.cls_list)

        # encoder_decoder
        self.encoder_decoder = EncoderDecoder(0)

        # postprocessor
        self.postprocessor = PostProcessor(0, len(self.cls_list))

        # loss builder
        self.loss_builder = LossBuilder(0)

        self.corner_loss = cfg.MODEL.FIRST_STAGE.CORNER_LOSS

        # head builder
        self.iou_loss = False
        self.heads = []
        head_cfg = cfg.MODEL.NETWORK.FIRST_STAGE.HEAD
        for i in range(len(head_cfg)):
            self.heads.append(HeadBuilder(self.batch_size, 
                self.anchor_builder.anchors_num, 0, head_cfg[i], is_training))
            if self.heads[-1].layer_type == 'IoU': self.iou_loss = True

        # target assigner
        self.target_assigner = TargetAssigner(0) # first stage

        self.vote_loss = False
        # layer builder
        layer_cfg = cfg.MODEL.NETWORK.FIRST_STAGE.ARCHITECTURE
        layers = []
        for i in range(len(layer_cfg)):
            layers.append(LayerBuilder(i, self.is_training, layer_cfg)) 
            if layers[-1].layer_type == 'Vote_Layer': self.vote_loss = True
        self.layers = layers

        self.attr_velo_loss = cfg.MODEL.FIRST_STAGE.PREDICT_ATTRIBUTE_AND_VELOCITY 

        self.__init_dict()

    def __init_dict(self):
        self.output = dict()
        # sampled xyz/feature
        self.output[maps_dict.KEY_OUTPUT_XYZ] = []
        self.output[maps_dict.KEY_OUTPUT_FEATURE] = []
        # generated anchors
        self.output[maps_dict.KEY_ANCHORS_3D] = [] # generated anchors
        # vote output
        self.output[maps_dict.PRED_VOTE_OFFSET] = []
        self.output[maps_dict.PRED_VOTE_BASE] = []
        # det output
        self.output[maps_dict.PRED_CLS] = []
        self.output[maps_dict.PRED_OFFSET] = []
        self.output[maps_dict.PRED_ANGLE_CLS] = []
        self.output[maps_dict.PRED_ANGLE_RES] = []
        self.output[maps_dict.CORNER_LOSS_PRED_BOXES_CORNERS] = []
        self.output[maps_dict.PRED_ATTRIBUTE] = []
        self.output[maps_dict.PRED_VELOCITY] = []
        # iou output
        self.output[maps_dict.PRED_IOU_3D_VALUE] = []
        # final result
        self.output[maps_dict.PRED_3D_BBOX] = []
        self.output[maps_dict.PRED_3D_SCORE] = []
        self.output[maps_dict.PRED_3D_CLS_CATEGORY] = []
        self.output[maps_dict.PRED_3D_ATTRIBUTE] = []
        self.output[maps_dict.PRED_3D_VELOCITY] = []
        self.output[maps_dict.PRED_POINT_SEG] = []


        self.prediction_keys = self.output.keys()
        
        self.labels = dict()
        self.labels[maps_dict.GT_CLS] = []
        self.labels[maps_dict.GT_OFFSET] = []
        self.labels[maps_dict.GT_ANGLE_CLS] = []
        self.labels[maps_dict.GT_ANGLE_RES] = []
        self.labels[maps_dict.GT_ATTRIBUTE] = []
        self.labels[maps_dict.GT_VELOCITY] = []
        self.labels[maps_dict.GT_BOXES_ANCHORS_3D] = []
        self.labels[maps_dict.GT_IOU_3D_VALUE] = []

        self.labels[maps_dict.GT_PMASK] = []
        self.labels[maps_dict.GT_NMASK] = []
        self.labels[maps_dict.CORNER_LOSS_GT_BOXES_CORNERS] = []
        self.labels[maps_dict.PL_LABEL_SEMSEGS] = []

    def network_forward(self, point_cloud, bn_decay, img_input, img_full_seg):
        l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
        l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,-1])

        num_point = l0_xyz.get_shape().as_list()[1]
        batch_size = l0_xyz.get_shape().as_list()[0]

        img_full_seg = tf.reshape(img_full_seg, [batch_size, 360, 1200, 1])
        pts2d = projection.tf_rect_to_image(tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3]),
                                            self.placeholders[maps_dict.PL_CALIB_P2])
        pts2d = tf.cast(pts2d, tf.int32)  # (B,N,2)
        indices = tf.concat([
            tf.expand_dims(tf.tile(tf.range(0, self.batch_size), [num_point]), axis=-1),  # (B*N, 1)
            tf.reshape(pts2d, [self.batch_size * num_point, 2])], axis=-1)  # (B*N,3)
        indices = tf.gather(indices, [0, 2, 1], axis=-1)  # image's shape is (y,x)
        img_full_seg = tf.reshape(tf.gather_nd(img_full_seg, indices),  # (B*N,C)
                                     [self.batch_size, num_point, -1])  # (B,N,C)

        nsamples = 256
        img_seg_npoints = 256
        pooling_size = []

        if self.cls_list[0] == 'Car':
            cls_int = 1
            pooling_size = [5.0, 1.7, 5.0]
        elif self.cls_list[0] == 'Pedestrian':
            cls_int = 2
            pooling_size = [1.2, 1.8, 1.2]
        elif  self.cls_list[0] == 'Cyclist':
            cls_int = 3
            pooling_size = [1.8, 1.8, 1.8]


        mask = tf.equal(img_full_seg, cls_int)
        mask = tf.reshape(mask, [self.batch_size, num_point])

        img_seg_masked, indices = tf_gather_object_pc(img_full_seg, mask, npoints=img_seg_npoints)
        img_seg_masked.set_shape([batch_size, img_seg_npoints, 1])

        img_seg_point_cloud = tf.gather_nd(l0_xyz, indices)
        img_seg_point_cloud.set_shape([batch_size, img_seg_npoints, 3])

        img_input = tf.image.resize_images(img_input, [360, 1200],
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

        xyz_list, feature_list, fps_idx_list = [l0_xyz], [l0_points], [None]

        point_seg_net = None

        for layer in self.layers:
            if layer.layer_type == 'Vote_Layer':
                l3_points = pointnet_fp_module(xyz_list[2], xyz_list[4], feature_list[2], feature_list[4], [256], layer.is_training, bn_decay, scope='fa_layer1')
                l2_points = pointnet_fp_module(xyz_list[1], xyz_list[2], feature_list[1], l3_points, [256], layer.is_training, bn_decay, scope='fa_layer2')
                l1_points = pointnet_fp_module(xyz_list[0], xyz_list[1], feature_list[0], l2_points, [256], layer.is_training, bn_decay, scope='fa_layer3')

                # net = tf_util.conv1d(l1_points, 128, 1, padding='VALID', bn=True,
                #                      is_training=layer.is_training, scope='img-seg-conv1d-fc1', bn_decay=bn_decay)
                # net = tf_util.dropout(net, keep_prob=0.7, is_training=layer.is_training, scope='img-seg-dp1')
                # logits = tf_util.conv1d(net, 2, 1, padding='VALID', activation_fn=None, scope='img-seg-conv1d-fc2')
                # self.output[maps_dict.PRED_POINT_SEG].append(logits)

                point_seg_net = tf.gather_nd(l1_points, indices)
                point_seg_net.set_shape([batch_size, img_seg_npoints, 256])


            xyz_list, feature_list, fps_idx_list = layer.build_layer(xyz_list, feature_list, fps_idx_list, bn_decay, self.output, self.placeholders[maps_dict.PL_CALIB_P2], img_input, img_seg_point_cloud, point_seg_net, pooling_size)




        cur_head_start_idx = len(self.output[maps_dict.KEY_OUTPUT_XYZ])
        for head in self.heads:
            head.build_layer(xyz_list, feature_list, bn_decay, self.output)
        merge_head_prediction(cur_head_start_idx, self.output, self.prediction_keys)


    def model_forward(self, bn_decay=None):
        points_input_det = self.placeholders[maps_dict.PL_POINTS_INPUT]
        img_input_det = self.placeholders[maps_dict.PL_IMG_INPUT]
        img_full_seg = self.placeholders[maps_dict.PL_IMG_FULL_SEG_INPUT]

        # forward the point cloud
        self.network_forward(points_input_det, bn_decay, img_input_det, img_full_seg)
 
        # generate anchors
        base_xyz = self.output[maps_dict.KEY_OUTPUT_XYZ][-1]
        anchors = self.anchor_builder.generate(base_xyz) # [bs, pts_num, 1/cls_num, 7]
        self.output[maps_dict.KEY_ANCHORS_3D].append(anchors)

        if self.is_training: # training mode
            self.train_forward(-1, anchors) 
        else: # testing mode
            self.test_forward(-1, anchors)


    def train_forward(self, index, anchors):
        """
        Calculating loss
        """
        base_xyz = self.output[maps_dict.KEY_OUTPUT_XYZ][index]
        pred_offset = self.output[maps_dict.PRED_OFFSET][index]
        pred_angle_cls = self.output[maps_dict.PRED_ANGLE_CLS][index]
        pred_angle_res = self.output[maps_dict.PRED_ANGLE_RES][index]

        gt_boxes_3d = self.placeholders[maps_dict.PL_LABEL_BOXES_3D]
        gt_classes = self.placeholders[maps_dict.PL_LABEL_CLASSES]
        gt_angle_cls = self.placeholders[maps_dict.PL_ANGLE_CLS]
        gt_angle_res = self.placeholders[maps_dict.PL_ANGLE_RESIDUAL]

        # point_sem_labels = self.placeholders[maps_dict.PL_LABEL_SEMSEGS]

        if maps_dict.PL_LABEL_ATTRIBUTES in self.placeholders.keys():
            gt_attributes = self.placeholders[maps_dict.PL_LABEL_ATTRIBUTES]
        else: gt_attributes = None

        if maps_dict.PL_LABEL_VELOCITY in self.placeholders.keys():
            gt_velocity = self.placeholders[maps_dict.PL_LABEL_VELOCITY]
        else: gt_velocity = None

        returned_list = self.target_assigner.assign(base_xyz, anchors, gt_boxes_3d, gt_classes, gt_angle_cls, gt_angle_res, gt_velocity, gt_attributes)

        assigned_idx, assigned_pmask, assigned_nmask, assigned_gt_boxes_3d, assigned_gt_labels, assigned_gt_angle_cls, assigned_gt_angle_res, assigned_gt_velocity, assigned_gt_attribute = returned_list

        # encode offset
        assigned_gt_offset, assigned_gt_angle_cls, assigned_gt_angle_res = self.encoder_decoder.encode(base_xyz, assigned_gt_boxes_3d, anchors)

        # corner_loss
        corner_loss_angle_cls = tf.cast(tf.one_hot(assigned_gt_angle_cls, depth=cfg.MODEL.ANGLE_CLS_NUM, on_value=1, off_value=0, axis=-1), tf.float32) # bs, pts_num, cls_num, -1
        pred_anchors_3d = self.encoder_decoder.decode(base_xyz, pred_offset, corner_loss_angle_cls, pred_angle_res, self.is_training, anchors) # [bs, points_num, cls_num, 7]
        pred_corners = transfer_box3d_to_corners(pred_anchors_3d) # [bs, points_num, cls_num, 8, 3] 
        gt_corners = transfer_box3d_to_corners(assigned_gt_boxes_3d) # [bs, points_num, cls_num,8,3]
        self.output[maps_dict.CORNER_LOSS_PRED_BOXES_CORNERS].append(pred_corners)
        self.labels[maps_dict.CORNER_LOSS_GT_BOXES_CORNERS].append(gt_corners)
        

        self.labels[maps_dict.GT_CLS].append(assigned_gt_labels)
        self.labels[maps_dict.GT_BOXES_ANCHORS_3D].append(assigned_gt_boxes_3d)
        self.labels[maps_dict.GT_OFFSET].append(assigned_gt_offset)
        self.labels[maps_dict.GT_ANGLE_CLS].append(assigned_gt_angle_cls)
        self.labels[maps_dict.GT_ANGLE_RES].append(assigned_gt_angle_res)
        self.labels[maps_dict.GT_ATTRIBUTE].append(assigned_gt_attribute)
        self.labels[maps_dict.GT_VELOCITY].append(assigned_gt_velocity)
        self.labels[maps_dict.GT_PMASK].append(assigned_pmask)
        self.labels[maps_dict.GT_NMASK].append(assigned_nmask)
        # self.labels[maps_dict.PL_LABEL_SEMSEGS].append(point_sem_labels)

        self.loss_builder.forward(index, self.labels, self.output, self.placeholders, self.corner_loss, self.vote_loss, self.attr_velo_loss, self.iou_loss)


    def test_forward(self, index, anchors):
        base_xyz = self.output[maps_dict.KEY_OUTPUT_XYZ][index]

        pred_cls = self.output[maps_dict.PRED_CLS][index] # [bs, points_num, cls_num + 1/0]
        pred_offset = self.output[maps_dict.PRED_OFFSET][index]
        pred_angle_cls = self.output[maps_dict.PRED_ANGLE_CLS][index]
        pred_angle_res = self.output[maps_dict.PRED_ANGLE_RES][index]

        # decode predictions
        pred_anchors_3d = self.encoder_decoder.decode(base_xyz, pred_offset, pred_angle_cls, pred_angle_res, self.is_training, anchors) # [bs, points_num, cls_num, 7]
        
        # decode classification
        if cfg.MODEL.FIRST_STAGE.CLS_ACTIVATION == 'Softmax':
            # softmax 
            pred_score = tf.nn.softmax(pred_cls)
            pred_score = tf.slice(pred_score, [0, 0, 1], [-1, -1, -1])
        else: # sigmoid
            pred_score = tf.nn.sigmoid(pred_cls)

        # using IoU branch proposed by sparse-to-dense
        if self.iou_loss:
            pred_iou = self.output[maps_dict.PRED_IOU_3D_VALUE][index]
            pred_score = pred_score * pred_iou

        if len(self.output[maps_dict.PRED_ATTRIBUTE]) <= 0:
            pred_attribute = None
        else: pred_attribute = self.output[maps_dict.PRED_ATTRIBUTE][index]

        if len(self.output[maps_dict.PRED_VELOCITY]) <= 0:
            pred_velocity = None
        else: pred_velocity = self.output[maps_dict.PRED_VELOCITY][index]

        self.postprocessor.forward(pred_anchors_3d, pred_score, self.output, pred_attribute, pred_velocity)

