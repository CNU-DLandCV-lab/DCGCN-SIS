import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, chevb_model, pointnet_fp_module, pointnet_upsample
from loss import *


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    sem_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, sem_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:, :, :3]
    l0_points = point_cloud[:, :, 3:]
    end_points['l0_xyz'] = l0_xyz

    # Shared encoder
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1',pooling='att_pooling')
    l2_xyz, l2_points, l2_indices = chevb_model(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32, scope='layer2',mlp=[64,64,128], bn_decay=bn_decay, output_dim=128, bn=True, pooling='att_pooling', is_training=is_training)
    l3_xyz, l3_points, l3_indices = chevb_model(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32, scope='layer3', mlp=[128,128,256], bn_decay=bn_decay, output_dim=256,bn=True, pooling='att_pooling',is_training=is_training)
    l4_xyz, l4_points, l4_indices = chevb_model(l3_xyz, l3_points, npoint=16, radius=0.8, nsample=32, scope='layer4', mlp=[256,256,512], bn_decay=bn_decay, output_dim=512,bn=True, pooling='att_pooling', is_training=is_training)
    
    # Semantic decoder
    l3_points_sem = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512,512], is_training, bn_decay, scope='sem_fa_layer1')
    l2_points_sem = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points_sem, [256,256], is_training, bn_decay, scope='sem_fa_layer2')
    l1_points_sem = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points_sem, [256,128], is_training, bn_decay, scope='sem_fa_layer3')
    l0_points_sem = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_sem, [128,128,128], is_training, bn_decay, scope='sem_fa_layer4')

    # Sem channel aggregation
    l2_points_sem_up = pointnet_upsample(l0_xyz, l2_xyz, l2_points_sem, scope='sem_up1')
    l1_points_sem_up = pointnet_upsample(l0_xyz, l1_xyz, l1_points_sem, scope='sem_up2')
    net_sem_0 = tf.add(tf.concat([l0_points_sem, l1_points_sem_up], axis=-1, name='sem_up_concat'), l2_points_sem_up,name='sem_up_add')
    net_sem_0 = tf_util.conv1d(net_sem_0, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='sem_fc1', bn_decay=bn_decay)

    # Instance decoder
    l3_points_ins = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512,512], is_training, bn_decay, scope='ins_fa_layer1')
    l2_points_ins = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points_ins, [256,256], is_training, bn_decay, scope='ins_fa_layer2')
    l1_points_ins = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points_ins, [256,128], is_training, bn_decay, scope='ins_fa_layer3')
    l0_points_ins = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_ins, [128,128,128], is_training, bn_decay, scope='ins_fa_layer4')

    # Ins channel aggregation
    l2_points_ins_up = pointnet_upsample(l0_xyz, l2_xyz, l2_points_ins, scope='ins_up1')
    l1_points_ins_up = pointnet_upsample(l0_xyz, l1_xyz, l1_points_ins, scope='ins_up2')
    net_ins_0 = tf.add(tf.concat([l0_points_ins, l1_points_ins_up], axis=-1, name='ins_up_concat'), l2_points_ins_up, name='ins_up_add')
    net_ins_0 = tf_util.conv1d(net_ins_0, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='ins_fc1', bn_decay=bn_decay)

    # Semantic-fused Instance 
    sem = tf.matmul(net_sem_0, tf.transpose(net_sem_0, perm=[0, 2, 1]))
    sim_sem = tf.multiply(net_sem_0, net_sem_0)
    sim_sem = tf.reduce_sum(sim_sem, 2, keep_dims=True)
    sim_sem = tf.sqrt(sim_sem)
    sim_sem = tf.matmul(sim_sem, tf.transpose(sim_sem, perm=[0, 2, 1]))
    sim_sem = tf.add(sim_sem, 1e-7)
    sem = tf.div(sem, sim_sem)
    tf.add_to_collection('sem-f-ins', sem)

    sem = tf.matmul(sem, net_sem_0)
    sem = tf.layers.dense(inputs=sem, units=128, activation=None, use_bias=False)
    sem = tf_util.batch_norm_for_conv1d(sem, is_training, bn_decay, "sem_bn", is_dist=False)
    sem = tf.nn.relu(sem)

    gate_sem = tf.layers.dense(inputs=sem, units=128, activation=None, use_bias=False)
    gate_sem = tf.nn.sigmoid(gate_sem)
    net_ins = tf.add(tf.multiply(bi_sem, gate_sem), tf.multiply(tf.subtract(tf.ones_like(gate_sem), gate_sem), net_ins_0))

    net_ins_2 = tf.concat([net_ins_0, sem], axis=-1, name='net_ins_2_concat')

    # Instance-fused Semantic
    net_ins_cache_0 = tf_util.conv1d(net_ins_2, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='ins_cache_1', bn_decay=bn_decay)
    net_ins_cache_1 = tf.reduce_mean(net_ins_cache_0, axis=1, keep_dims=True, name='ins_cache_2')
    net_ins_cache_1 = tf.tile(net_ins_cache_1, [1, num_point, 1], name='ins_cache_tile')

    net_sem_1 = net_sem_0 + net_ins_cache_1
    net_sem_2 = tf.concat([net_sem_0, net_sem_1], axis=-1, name='net_sem_2_concat')


    # Output
    net_sem_3 = tf_util.conv1d(net_sem_2, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='sem_cache_3', bn_decay=bn_decay)
    net_sem_4 = net_sem_3 + net_sem_1
    net_sem_5 = tf_util.conv1d(net_sem_4, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='sem_fc2', bn_decay=bn_decay)
    net_sem_6 = tf_util.dropout(net_sem_5, keep_prob=0.5, is_training=is_training, scope='sem_dp_4')
    net_sem_6 = tf_util.conv1d(net_sem_6, num_class, 1, padding='VALID', activation_fn=None, scope='sem_fc5')


    net_ins_3 = tf_util.conv1d(net_ins_2, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='ins_fc2', bn_decay=bn_decay)
    net_ins_4 = net_ins_3 + net_ins + net_sem_3
    net_ins_5 = tf_util.conv1d(net_ins_4, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='ins_cache_3', bn_decay=bn_decay)
    net_ins_6 = tf_util.dropout(net_ins_5, keep_prob=0.5, is_training=is_training, scope='ins_dp_5')
    net_ins_6 = tf_util.conv1d(net_ins_6, 5, 1, padding='VALID', activation_fn=None, scope='ins_fc5')

    return net_sem_6, net_ins_6


def get_loss(pred, ins_label, pred_sem_label, pred_sem, sem_label):
    """ pred:   BxNxE,
        ins_label:  BxN
        pred_sem_label: BxN
        pred_sem: BxNx13
        sem_label: BxN
    """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=sem_label, logits=pred_sem)
    tf.summary.scalar('classify loss', classify_loss)

    feature_dim = pred.get_shape()[-1]
    delta_v = 0.5
    delta_d = 1.5
    param_var = 1.
    param_dist = 1.
    param_reg = 0.001
    
    disc_loss, l_var, l_dist, l_reg = discriminative_loss(pred, ins_label, feature_dim, delta_v, delta_d, param_var, param_dist, param_reg)

    loss = classify_loss + disc_loss

    tf.add_to_collection('losses', loss)
    return loss, classify_loss, disc_loss, l_var, l_dist, l_reg

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
