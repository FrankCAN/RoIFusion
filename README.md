# RoIFusion: 3D Object Detection from LiDAR and Vision
created by Can Chen, Luca Zanotti Fragonara, Antonios Tsourdos from Cranfield University

[[Paper]](https://arxiv.org/abs/2009.04554)

# Overview
We would like to leverage the advantages of LIDAR and camera sensors by proposing a deep neural network architecture for the fusion and the efficient detection of 3D objects by identifying their corresponding 3D bounding boxes with orientation. In order to achieve this task, instead of densely combining the point-wise feature of the point cloud and the related pixel features, we propose a novel fusion algorithm by projecting a set of 3D Region of Interests (RoIs) from the point clouds to the 2D RoIs of the corresponding the images.

# Requirement
* [TensorFlow](https://www.tensorflow.org/)
