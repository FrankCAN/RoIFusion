import tensorflow as tf
import numpy as np

import utils.tf_util as tf_util
import projection
import resnet_model

from core.config import cfg
from utils.layers_util import *

import dataset.maps_dict as maps_dict

class LayerBuilder:
    def __init__(self, layer_idx, is_training, layer_cfg):
        self.layer_idx = layer_idx
        self.is_training = is_training

        self.layer_architecture = layer_cfg[self.layer_idx]

        self.xyz_index = self.layer_architecture[0]
        self.feature_index = self.layer_architecture[1]
        self.radius_list = self.layer_architecture[2]
        self.nsample_list = self.layer_architecture[3]
        self.mlp_list = self.layer_architecture[4]
        self.bn = self.layer_architecture[5]

        self.fps_sample_range_list = self.layer_architecture[6]
        self.fps_method_list = self.layer_architecture[7]
        self.npoint_list = self.layer_architecture[8]
        assert len(self.fps_sample_range_list) == len(self.fps_method_list)
        assert len(self.fps_method_list) == len(self.npoint_list)

        self.former_fps_idx = self.layer_architecture[9]
        self.use_attention = self.layer_architecture[10]
        self.layer_type = self.layer_architecture[11]
        self.scope = self.layer_architecture[12] 
        self.dilated_group = self.layer_architecture[13]
        self.vote_ctr_index = self.layer_architecture[14]
        self.aggregation_channel = self.layer_architecture[15]

        if self.layer_type in ['SA_Layer', 'Vote_Layer', 'SA_Layer_SSG_Last']:
            assert len(self.xyz_index) == 1
        elif self.layer_type == 'FP_Layer':
            assert len(self.xyz_index) == 2
        else: raise Exception('Not Implementation Error!!!')

    def build_layer(self, xyz_list, feature_list, fps_idx_list, bn_decay, output_dict, p2, img_input=None, img_seg_point_cloud=None, point_seg_net=None, pooling_size=None):
        """
        Build layers
        """
        xyz_input = []
        for xyz_index in self.xyz_index:
            xyz_input.append(xyz_list[xyz_index])
 
        feature_input = []
        for feature_index in self.feature_index:
            feature_input.append(feature_list[feature_index])

        if self.former_fps_idx != -1:
            former_fps_idx = fps_idx_list[self.former_fps_idx]
        else:
            former_fps_idx = None

        if self.vote_ctr_index != -1:
            vote_ctr = xyz_list[self.vote_ctr_index]
        else:
            vote_ctr = None


        # image feature extraction
        if vote_ctr is not None:
            num_point = vote_ctr.get_shape().as_list()[1]
            batch_size = vote_ctr.get_shape().as_list()[0]

            img_rois = projection.crop_rois(vote_ctr, p2, img_input, pooling_size=pooling_size)

            img_features = tf_util.conv2d(img_rois, 256, [3, 3], padding='VALID', stride=[1, 1], bn=self.bn,
                                          is_training=self.is_training, scope=self.scope, bn_decay=bn_decay)
            img_features = tf.reshape(img_features, [batch_size, num_point, 256])

        else:
            img_features = None






        if self.layer_type == 'SA_Layer':
            if self.scope == 'layer4':
                new_xyz, new_points, new_fps_idx = pointnet_sa_module_msg(
                    xyz_input[0], feature_input[0],
                    self.radius_list, self.nsample_list,
                    self.mlp_list, self.is_training, bn_decay, self.bn,
                    self.fps_sample_range_list, self.fps_method_list, self.npoint_list,
                    former_fps_idx, self.use_attention, self.scope,
                    self.dilated_group, vote_ctr, self.aggregation_channel, img_features=img_features)
            else:
                new_xyz, new_points, new_fps_idx = pointnet_sa_module_msg(
                    xyz_input[0], feature_input[0],
                    self.radius_list, self.nsample_list,
                    self.mlp_list, self.is_training, bn_decay, self.bn,
                    self.fps_sample_range_list, self.fps_method_list, self.npoint_list,
                    former_fps_idx, self.use_attention, self.scope,
                    self.dilated_group, vote_ctr, self.aggregation_channel)
            xyz_list.append(new_xyz)
            feature_list.append(new_points)
            fps_idx_list.append(new_fps_idx)

        elif self.layer_type == 'SA_Layer_SSG_Last':
            new_points = pointnet_sa_module(
                xyz_input[0], feature_input[0],
                self.mlp_list, self.is_training, bn_decay,
                self.bn, self.scope,
            )
            xyz_list.append(None)
            feature_list.append(new_points)
            fps_idx_list.append(None)

        elif self.layer_type == 'FP_Layer':
            new_points = pointnet_fp_module(xyz_input[0], xyz_input[1], feature_input[0], feature_input[1], self.mlp_list, self.is_training, bn_decay, self.scope, self.bn)
            xyz_list.append(xyz_input[0])
            feature_list.append(new_points)
            fps_idx_list.append(None)
        
        elif self.layer_type == 'Vote_Layer':
            # new_xyz, new_points, ctr_offsets = vote_layer(xyz_input[0], feature_input[0], self.mlp_list, self.is_training, bn_decay, self.bn, self.scope)
            # output_dict[maps_dict.PRED_VOTE_BASE].append(xyz_input[0])
            # output_dict[maps_dict.PRED_VOTE_OFFSET].append(ctr_offsets)
            #
            # if voting_xyz is not None:
            #     voting_xyz_update = tf.concat([new_xyz, voting_xyz], axis=1)
            #     voting_feature_update = tf.concat([new_points, voting_points], axis=1)
            #     new_xyz, new_points, fps_idx_update = pointnet_fps_method(voting_xyz_update, voting_feature_update,
            #                                                               [-1], ['F-FPS'], [256], self.scope, vote_ctr=None)

            xyz_update = tf.concat([xyz_input[0], img_seg_point_cloud], axis=1)
            feature_update = tf.concat([feature_input[0], point_seg_net], axis=1)

            xyz_update, feature_update, fps_idx_update = pointnet_fps_method(xyz_update, feature_update,
                                                                      [-1], ['F-FPS'], [256], self.scope, vote_ctr=None)

            new_xyz, new_points, ctr_offsets = vote_layer(xyz_update, feature_update, self.mlp_list, self.is_training, bn_decay, self.bn, self.scope)
            output_dict[maps_dict.PRED_VOTE_BASE].append(xyz_update)
            output_dict[maps_dict.PRED_VOTE_OFFSET].append(ctr_offsets)


            xyz_list.append(new_xyz)
            feature_list.append(new_points)
            fps_idx_list.append(None)

        return xyz_list, feature_list, fps_idx_list
         
