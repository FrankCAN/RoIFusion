import tensorflow as tf
from modeling_util import get_box3d_corners_helper
import numpy as np

def tf_rect_to_image(pts3d, calib):
    """
    Projects 3D points into image space

    Args:
        pts3d: a tensor of rect points in the shape [B, N, 3].
        calib: tensor [3, 4] stereo camera calibration p2 matrix
    Returns:
        pts2d: a float32 tensor points in image space -
            B x N x [x, y]
    """
    B = pts3d.shape[0]
    N = pts3d.shape[1]
    calib_expand = tf.tile(tf.expand_dims(calib, 1), [1,N,1,1]) # (B,N,3, 4)
    # pts3d_list = tf.reshape(B*N, 3)
    pts3d_hom = tf.concat([pts3d, tf.ones((B,N,1))], axis=-1) # (B,N,4)
    pts3d_hom = tf.expand_dims(pts3d_hom, axis=-1) # (B,N,4,1)
    pts2d_hom = tf.matmul(calib_expand, pts3d_hom) # (B,N,3,1)
    pts2d_hom = tf.squeeze(pts2d_hom, axis=-1) # (B,N,3)
    depth = tf.gather(pts2d_hom, 2, axis=-1)
    return tf.stack([
        tf.gather(pts2d_hom, 0, axis=-1)/depth,
        tf.gather(pts2d_hom, 1, axis=-1)/depth,
    ], axis=-1)

def tf_project_to_image_space(boxes, calib, image_shape, pooling_size):
    """
    Projects 3D tensor boxes into image space

    Args:
        boxes: a tensor of anchors in the shape [B, 7].
            The anchors are in the format [x, y, z, l, h, w, ry]
        calib: tensor [3, 4] stereo camera calibration p2 matrix
        image_shape: a float32 tensor of shape [2]. This is dimension of
            the image [h, w]

    Returns:
        box_corners: a float32 tensor corners in image space -
            N x [x1, y1, x2, y2]
        box_corners_norm: a float32 tensor corners as a percentage
            of the image size - N x [x1, y1, x2, y2]

    """
    boxes_reshape = tf.reshape(boxes, [-1, 3])
    # dim = np.tile([[3.0, 1.6, 3.0]], (pts_size, 1))
    # lhw = tf.ones_like(boxes_reshape) * 1.6
    lhw = tf.tile([pooling_size], (boxes_reshape.shape[0], 1))
    ry_ = tf.zeros([boxes_reshape.shape[0], 1])

    calib_ = tf.tile(tf.expand_dims(calib, 1), [1, boxes.shape[1], 1, 1])
    calib = tf.reshape(calib_, [-1, 3, 4])

    boxes = tf.concat([boxes_reshape, lhw, ry_], axis=-1)




    batch_size = boxes.shape[0]
    box_center = tf.slice(boxes, [0,0], [-1, 3])
    box_size = tf.slice(boxes, [0,3], [-1, 3])
    box_angle = tf.slice(boxes, [0,6], [-1, 1])
    corners_3d = get_box3d_corners_helper(
        box_center, tf.gather(box_angle, 0, axis=-1), box_size) # (B,8,3)
    #corners_3d_list = tf.reshape(corners_3d, [batch_size*8, 3])
    # corners_3d = tf.expand_dims(corners_3d, axis=2) # (B,8,1,3)
    corners_3d_hom = tf.concat([corners_3d, tf.ones((batch_size,8,1))], axis=-1) # (B,8,4)
    corners_3d_hom = tf.expand_dims(corners_3d_hom, axis=-1) # (B,8,4,1)
    # calib_tiled = tf.tile(tf.expand_dims(calib, 1), [1,8,1,1]) # (B,8,3,4)
    calib_tiled = tf.tile(tf.expand_dims(calib, 1), [1, 8, 1, 1])  # (B,8,3,4)
    projected_pts = tf.matmul(calib_tiled, corners_3d_hom) # (B,8,3,1)
    projected_pts = tf.squeeze(projected_pts, axis=-1) # (B,8,3)

    projected_pts_norm = projected_pts/tf.slice(projected_pts, [0,0,2], [-1,-1,1]) # divided by depth

    corners_2d = tf.gather(projected_pts_norm, [0,1], axis=-1) # (B,8,2)

    pts_2d_min = tf.reduce_min(corners_2d, axis=1)
    pts_2d_max = tf.reduce_max(corners_2d, axis=1) # (B, 2)
    box_corners = tf.stack([
        tf.maximum(tf.gather(pts_2d_min, 0, axis=1), 0),
        tf.maximum(tf.gather(pts_2d_min, 1, axis=1), 0),
        tf.minimum(tf.gather(pts_2d_max, 0, axis=1), 1200),
        tf.minimum(tf.gather(pts_2d_max, 1, axis=1), 360),
        ], axis=1) # (B,4) (x1, y1, x2, y2)

    # Normalize
    image_shape_h = image_shape[0]
    image_shape_w = image_shape[1]

    image_shape_tiled = tf.tile([[image_shape_w, image_shape_h, image_shape_w, image_shape_h]], [batch_size,1])

    box_corners_norm = box_corners / tf.to_float(image_shape_tiled)

    return box_corners, box_corners_norm


def crop_rois(ctr_points, p2, img_input, crop_size=[3, 3], pooling_size=[5.0, 1.7, 5.0]):
    _img_pixel_size = np.asarray([360, 1200])
    num_point = ctr_points.get_shape().as_list()[1]
    batch_size = ctr_points.get_shape().as_list()[0]

    box2d_corners, box2d_corners_norm = tf_project_to_image_space(ctr_points, p2, _img_pixel_size, pooling_size)

    box2d_corners_norm_reorder = tf.stack([tf.gather(box2d_corners_norm, 1, axis=-1), tf.gather(box2d_corners_norm, 0, axis=-1),
                                           tf.gather(box2d_corners_norm, 3, axis=-1), tf.gather(box2d_corners_norm, 2, axis=-1)], axis=-1)

    num_boxes = tf.concat([tf.ones([num_point]) * i for i in np.arange(batch_size)], axis=0)
    num_boxes = tf.cast(num_boxes, tf.int32)
    img_rois = tf.image.crop_and_resize(img_input, box2d_corners_norm_reorder, num_boxes, crop_size)

    return img_rois