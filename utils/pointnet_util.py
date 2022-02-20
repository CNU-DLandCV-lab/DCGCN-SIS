"""
Modify based on PointNet++ Layers (Author: Charles R. Qi. Date: November 2017)

"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling_nd'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
import spec_graph_util
from spec_graph_util import spec_conv2d
from spec_graph_util import spec_conv2d_modul
from spec_graph_util import spec_hier_cluster_pool

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz

def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='att_pooling', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)
        elif pooling == 'att_pooling':
            batch_size = tf.shape(new_points)[0]
            num_points = tf.shape(new_points)[1]
            num_neigh = tf.shape(new_points)[2]
            d = new_points.get_shape()[3].value
            f_reshaped = tf.reshape(new_points, shape=[-1, num_neigh, d])
            att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name='fc')
            att_scores = tf.nn.softmax(att_activation, axis=1)
            f_agg = f_reshaped * att_scores
            f_agg = tf.reduce_sum(f_agg, axis=1)
            f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
            new_points = tf_util.conv2d(f_agg, num_out_channel, [1, 1], 'mlp', stride=[1, 1], padding='VALID',
                                        is_training=is_training, bn_decay=bn_decay)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,2,3,1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat

def chevb_model(xyz, points, npoint, radius, nsample,scope ,mlp, bn_decay,output_dim,bn=True, pooling='max',is_training=True,k=5):
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope("sample_and_group") as sc:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, True, True)
        # grouped_xyz this is local xyz wrt to each center point
        # dim B N K 3
        # controls using what feature for covariance computation
        with tf.variable_scope("Laplacian") as sc:
            local_cord = grouped_xyz
            in_shape = new_points.get_shape().as_list()
            W = spec_graph_util.get_adj_mat_dist_euclidean(local_cord[:,:,:,0:3] , flag_normalized = True)
            W = tf.identity(W, name='adjmat')
            W_knn = spec_graph_util.cov_mat_k_nn_graph(W, k = 32 )
            W_knn = spec_graph_util.corv_mat_setdiag_zero(W_knn)
            W_knn = tf.identity(W_knn, name='adjmat_knn')
            L = spec_graph_util.corv_mat_laplacian0(W_knn , flag_normalized = True)
            L = tf.identity(L, name='laplacian')
            
        with tf.variable_scope("MLP") as sc:
            for i, num_out_channel in enumerate(mlp):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)

            x=tf.matmul(L,new_points)
            # gcn
        with tf.variable_scope("GCN") as sc:

            W_spec=tf.get_variable('weights_spec',[x.shape[-1],output_dim],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points_=tf.matmul(x1,W_spec)
            new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],output_dim))
            new_points_=prelu(new_points_)

            #attentive pooling  (Original: RandLA-Net)
            batch_size = tf.shape(new_points)[0]
            num_points = tf.shape(new_points)[1]
            num_neigh = tf.shape(new_points)[2]
            d = new_points.get_shape()[3].value
            f_reshaped = tf.reshape(new_points, shape=[-1, num_neigh, d])
            att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name='fc')
            att_scores = tf.nn.softmax(att_activation, axis=1)
            f_agg = f_reshaped * att_scores
            f_agg = tf.reduce_sum(f_agg, axis=1)
            f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
            new_points = tf_util.conv2d(f_agg, num_out_channel, [1, 1], 'mlp', stride=[1, 1], padding='VALID', is_training=is_training, bn_decay=bn_decay)
            new_points_ = tf.squeeze(new_points, [2])

    return new_xyz, new_points_, idx


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1


def query_and_group_points(xyz, points, new_xyz, nsample, radius, knn=False,
                           use_xyz=True, normalize_radius=True, orientations=None):

    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
        pts_cnt = nsample  # Hack. By right should make sure number of input points < nsample
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)

    tf.summary.histogram('pts_cnt', pts_cnt)

    # Group XYZ coordinates
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz = grouped_xyz - tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
    if normalize_radius:
        grouped_xyz /= radius  # Scale normalization
    # 2D-rotate via orientations if necessary
    if orientations is not None:
        cosval = tf.expand_dims(tf.cos(orientations), axis=2)
        sinval = tf.expand_dims(tf.sin(orientations), axis=2)
        grouped_xyz = tf.stack([cosval * grouped_xyz[:, :, :, 0] + sinval * grouped_xyz[:, :, :, 1],
                                -sinval * grouped_xyz[:, :, :, 0] + cosval * grouped_xyz[:, :, :, 1],
                                grouped_xyz[:, :, :, 2]], axis=3)

    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:

            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_points, idx

def pointnet_upsample(xyz1, xyz2, points2, scope):
    """ PointNet Feature Propogation (FP) Module
            Input:
                xyz1: (batch_size, ndataset1, 3) TF tensor
                xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
                points2: (batch_size, ndataset2, nchannel2) TF tensor
            Return:
                new_points: (batch_size, ndataset1, nchannel2) TF tensor
    """
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)  # B x ndataset1 x nchannel2

        return interpolated_points


def prelu(_x, name=None):
    """parametric ReLU activation"""
    if name is None:
        name ="alpha"
    _alpha = tf.get_variable(name, shape=_x.get_shape()[-1], dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
    pos = tf.nn.relu(_x)
    neg = _alpha * (_x - tf.abs(_x)) * 0.5

    return pos + neg