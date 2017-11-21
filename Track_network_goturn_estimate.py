import os
import cv2
import math
import glob
import random
import skvideo.io
import numpy as np
import tensorflow as tf
from random import shuffle
from time import gmtime, strftime

from load_alov import video
from load_alov import frame
from load_alov import BoundingBox
from load_alov import annotation
from load_alov import cropPadImage
from load_alov import preprocess
from load_alov import load_annotation_file

model_name = 'model64_4l_2017-11-190954_re2.ckpt'

# Parameters
fisrt_file_index =3
end_file_index = fisrt_file_index+1

fisrt_file_index_vot =11
end_file_index_vot = fisrt_file_index_vot#+1
frame_step_vot = 2

initail_frame = 1
end_frame = 600

width_height_tunning = 1
x_y_startcrop_tunning = 1
# Number of samples to calculate validation and accuracy

# Network Parameters
n_output = 4  # Cx,Cy,W,H
dropout = 0.5  # Dropout, probability to keep units

#Training resize image size
width_resize = 227
height_resize = 227
channel_resize = 3

#weight and bias initial parameter
mu = 0
sigma = 0.01

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)#group means we split the input  into 'group' groups along the third demention
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups,3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

def conv_net(x_cur,x_pre, dropout):
    with tf.name_scope('current_frame_layers'):
        #conv1
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        with tf.name_scope('conv_1'):
            k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
            with tf.name_scope('weights'):
                conv1W = tf.Variable(tf.random_normal([k_h, k_w, 3, c_o], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                conv1b = tf.Variable(tf.random_normal([c_o], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                conv1_in = conv(x_cur, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1)
            conv1 = tf.nn.relu(conv1_in)

            with tf.name_scope('maxpool'):
                #maxpool1
                #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
                k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
                maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
            with tf.name_scope('lrn'):
                #lrn1
                #lrn(2, 2e-05, 0.75, name='norm1')
                radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
                lrn1 = tf.nn.local_response_normalization(maxpool1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)

        with tf.name_scope('conv_2'):
            #conv2
            #conv(5, 5, 256, 1, 1, group=2, name='conv2')
            k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group =2
            with tf.name_scope('weights'):
                conv2W = tf.Variable(tf.random_normal([k_h, k_w, 48, c_o], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                conv2b = tf.Variable(tf.random_normal([c_o], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                conv2_in = conv(lrn1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv2 = tf.nn.relu(conv2_in)

            with tf.name_scope('maxpool'):
                #maxpool2
                #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
                k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
                maxpool2 = tf.nn.max_pool(conv2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            with tf.name_scope('lrn'):
                #lrn2
                #lrn(2, 2e-05, 0.75, name='norm2')
                radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
                lrn2 = tf.nn.local_response_normalization(maxpool2 ,
                                                                  depth_radius=radius,
                                                                  alpha=alpha,
                                                                  beta=beta,
                                                                  bias=bias)
        with tf.name_scope('conv_3'):
            #conv3
            #conv(3, 3, 384, 1, 1, name='conv3')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
            with tf.name_scope('weights'):
                conv3W = tf.Variable(tf.random_normal([k_h, k_w, 256, c_o], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                conv3b = tf.Variable(tf.random_normal([c_o], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                conv3_in = conv(lrn2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv3 = tf.nn.relu(conv3_in)

        with tf.name_scope('conv_4'):
            #conv4
            #conv(3, 3, 384, 1, 1, group=2, name='conv4')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
            with tf.name_scope('weights'):
                conv4W = tf.Variable(tf.random_normal([k_h, k_w, 192, c_o], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                conv4b = tf.Variable(tf.random_normal([c_o], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv4 = tf.nn.relu(conv4_in)

        with tf.name_scope('conv_5'):
            #conv5
            #conv(3, 3, 256, 1, 1, group=2, name='conv5')
            k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
            with tf.name_scope('weights'):
                conv5W = tf.Variable(tf.random_normal([k_h, k_w, 192, c_o], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                conv5b = tf.Variable(tf.random_normal([c_o], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv5 = tf.nn.relu(conv5_in)

            with tf.name_scope('maxpool'):
                #maxpool5
                #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
                k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
                maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            with tf.name_scope('conv_reshape'):
                # Fully connected layer - 6*6*256 to 1024*4
                fcrw = tf.Variable(tf.random_normal([6*6*256, 1024*4], mean = mu, stddev = sigma))
                fc1 = tf.reshape(maxpool5, [-1, fcrw.get_shape().as_list()[0]])
###############################################################################
    with tf.name_scope('previous_frame_layers'):
        with tf.name_scope('conv_1'):
            #conv1
            #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
            k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
            with tf.name_scope('weights'):
                conv1W_p = tf.Variable(tf.random_normal([k_h, k_w, 3, c_o], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                conv1b_p = tf.Variable(tf.random_normal([c_o], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                conv1_in_p = conv(x_pre, conv1W_p, conv1b_p, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1)
            conv1_p = tf.nn.relu(conv1_in_p)

            with tf.name_scope('maxpool'):
                #maxpool1
                #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
                k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
                maxpool1_p = tf.nn.max_pool(conv1_p, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            with tf.name_scope('lrn'):
                #lrn1
                #lrn(2, 2e-05, 0.75, name='norm1')
                radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
                lrn1_p = tf.nn.local_response_normalization(maxpool1_p,
                                                              depth_radius=radius,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              bias=bias)
        with tf.name_scope('conv_2'):
            #conv2
            #conv(5, 5, 256, 1, 1, group=2, name='conv2')
            k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group =2
            with tf.name_scope('weights'):
                conv2W_p = tf.Variable(tf.random_normal([k_h, k_w, 48, c_o], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                conv2b_p = tf.Variable(tf.random_normal([c_o], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                conv2_in_p = conv(lrn1_p, conv2W_p, conv2b_p, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv2_p = tf.nn.relu(conv2_in_p)

            with tf.name_scope('maxpool'):
                #maxpool2
                #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
                k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
                maxpool2_p = tf.nn.max_pool(conv2_p, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            with tf.name_scope('lrn'):
                #lrn2
                #lrn(2, 2e-05, 0.75, name='norm2')
                radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
                lrn2_p = tf.nn.local_response_normalization(maxpool2_p ,
                                                              depth_radius=radius,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              bias=bias)

        with tf.name_scope('conv_3'):
            #conv3
            #conv(3, 3, 384, 1, 1, name='conv3')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
            with tf.name_scope('weights'):
                conv3W_p = tf.Variable(tf.random_normal([k_h, k_w, 256, c_o], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                conv3b_p = tf.Variable(tf.random_normal([c_o], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                conv3_in_p = conv(lrn2_p, conv3W_p, conv3b_p, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv3_p = tf.nn.relu(conv3_in_p)

        with tf.name_scope('conv_4'):
            #conv4
            #conv(3, 3, 384, 1, 1, group=2, name='conv4')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
            with tf.name_scope('weights'):
                conv4W_p = tf.Variable(tf.random_normal([k_h, k_w, 192, c_o], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                conv4b_p = tf.Variable(tf.random_normal([c_o], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                conv4_in_p = conv(conv3_p, conv4W_p, conv4b_p, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv4_p = tf.nn.relu(conv4_in_p)

        with tf.name_scope('conv_5'):
            #conv5
            #conv(3, 3, 256, 1, 1, group=2, name='conv5')
            k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
            with tf.name_scope('weights'):
                conv5W_p = tf.Variable(tf.random_normal([k_h, k_w, 192, c_o], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                conv5b_p = tf.Variable(tf.random_normal([c_o], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                conv5_in_p = conv(conv4_p, conv5W_p, conv5b_p, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv5_p = tf.nn.relu(conv5_in_p)

            with tf.name_scope('maxpool'):
                #maxpool5
                #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
                k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
                maxpool5_p = tf.nn.max_pool(conv5_p, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            with tf.name_scope('conv_reshape'):
                # Fully connected layer - 7*7*64 to 1024*4
                fcrw_p = tf.Variable(tf.random_normal([6*6*256, 1024*4], mean = mu, stddev = sigma))
                fc_p1 = tf.reshape(maxpool5_p, [-1, fcrw_p.get_shape().as_list()[0]])

###############################################################################
    with tf.name_scope('fullconect_layers'):
        with tf.name_scope('fc_1'):
            #Combine two frame into one full connect layer
            with tf.name_scope('weights'):
                fc1W = tf.Variable(tf.random_normal([6*6*256, 1024*4], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                fc1b = tf.Variable(tf.random_normal([1024*4], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                fc1 = tf.add(tf.matmul(fc1, fc1W), fc1b)

            with tf.name_scope('weights'):
                fc1W_p = tf.Variable(tf.random_normal([6*6*256, 1024*4], mean = mu, stddev = sigma))

            with tf.name_scope('out_sum'):
                fc_p1 = tf.matmul(fc_p1, fc1W_p)
                fc_d1_temp = tf.add(fc1, fc_p1)

            fc_d1 = tf.nn.relu(fc_d1_temp)

            with tf.name_scope('dropout'):
                fc_d1 = tf.nn.dropout(fc_d1, dropout)

        with tf.name_scope('fc_2'):
            # Fully connected layer - 7*7*64 to 1024*4
            with tf.name_scope('weights'):
                fc2W = tf.Variable(tf.random_normal([1024*4, 1024*4], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                fc2b = tf.Variable(tf.random_normal([1024*4], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                fc_d2 = tf.add(tf.matmul(fc_d1, fc2W), fc2b)

            fc_d2 = tf.nn.relu(fc_d2)

            with tf.name_scope('dropout'):
                fc_d2 = tf.nn.dropout(fc_d2, dropout)

        with tf.name_scope('fc_3'):
            # Fully connected layer - 7*7*64 to 1024*4
            with tf.name_scope('weights'):
                fc3W = tf.Variable(tf.random_normal([1024*4, 1024*4], mean = mu, stddev = sigma))
            with tf.name_scope('biases'):
                fc3b = tf.Variable(tf.random_normal([1024*4], mean = mu, stddev = sigma))
            with tf.name_scope('out'):
                fc_d3 = tf.add(tf.matmul(fc_d2, fc3W), fc3b)

            fc_d3 = tf.nn.relu(fc_d3)

            with tf.name_scope('dropout'):
                fc_d3 = tf.nn.dropout(fc_d3, dropout)

    with tf.name_scope('outputs'):
        # Output Layer - class prediction - 1024 to 4
        with tf.name_scope('weights'):
            outW = tf.Variable(tf.random_normal([1024*4, n_output], mean = mu, stddev = sigma))
        with tf.name_scope('biases'):
            outb = tf.Variable(tf.random_normal([n_output], mean = mu, stddev = sigma))
        with tf.name_scope('out'):
            out = tf.add(tf.matmul(fc_d3, outW), outb)

    return out

# define tf Graph input
with tf.name_scope('inputs'):
    x_cur = tf.placeholder(tf.float32, [None, width_resize, height_resize, 3], name='x_cur')
    x_pre = tf.placeholder(tf.float32, [None, width_resize, height_resize, 3], name='x_pre')
    #display concatenate input image in TensorBoard
    x_cur_pre = tf.concat([x_cur,x_pre],1)
    # tf.summary.image('x_cur_pre', x_cur_pre, 30)

    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

with tf.name_scope('truth'):
    y = tf.placeholder(tf.float32, [None, n_output], name='y_input')

with tf.name_scope('network'):
    with tf.name_scope('logits'):
        # Model
        logits = conv_net(x_cur, x_pre, keep_prob)
        tf.summary.histogram('logits', logits)

    # Define loss and optimizer
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.square(logits-y))
        cost_L1 = tf.reduce_mean(tf.square(logits-y)/2)
        #Send to TensorBoard to display
        tf.summary.scalar('cost', cost)
        tf.summary.scalar('cost_L1', cost_L1)

    #Accuracy
    with tf.name_scope('predict'):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #Send to TensorBoard to display
        tf.summary.scalar('Accuracy', accuracy)

    with tf.name_scope('initial'):
        # Initializing the variables
        init = tf.initialize_all_variables()
        # init = tf.global_variables_initializer()

#Create a saver object which will save all the variables
saver = tf.train.Saver()
#######################################################################

#add one more dimension
image_orginal_curr = []
image_orginal_prev = []

batch_x_prev =[]
batch_x_curr = []
batch_gt_prev = []
batch_y_curr = []

for file_num in range(fisrt_file_index, end_file_index):
    # file_num = 2
    # text_file = open("./Track_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    text_file = open("./Track_human_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    # text_file = open("./Track_test/test_video000"+format(file_num, '02d')+"/test_video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    groundtruth_frame = text_file.readlines()
    text_file.close()

    frame_len = len(groundtruth_frame)
    print 'len(groundtruth_frame):', frame_len

    end_frame = frame_len-1 - 6

    for num_img in range(1, frame_len):
        y_temp_cur = groundtruth_frame[num_img]
        y_temp_cur = y_temp_cur.split()
        y_temp_float_cur = [float(i) for i in y_temp_cur]

        y_temp_pre = groundtruth_frame[num_img-1]
        y_temp_pre = y_temp_pre.split()
        y_temp_float_pre = [float(i) for i in y_temp_pre]

        frame_index_cur = int(y_temp_float_cur[0])
        frame_index_pre = int(y_temp_float_pre[0])

        print 'file_num', file_num
        print 'frame_index_cur', frame_index_cur , ' frame_index_pre', frame_index_pre

        # image_path_cur = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_path_cur = './Track_human_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        # image_path_cur = './Track_test/test_video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_curr = cv2.imread(image_path_cur)
        image_orginal_curr.append(image_curr)

        # image_path_pre = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_path_pre = './Track_human_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        # image_path_pre = './Track_test/test_video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_prev = cv2.imread(image_path_pre)
        image_orginal_prev.append(image_prev)

        #Use previous frame to crop previous frame and current frame, and use current y as turth value
        x_value_pre = [y_temp_float_pre[1],y_temp_float_pre[3],y_temp_float_pre[5],y_temp_float_pre[7]]
        y_value_pre = [y_temp_float_pre[2],y_temp_float_pre[4],y_temp_float_pre[6],y_temp_float_pre[8]]
        x_value_cur = [y_temp_float_cur[1],y_temp_float_cur[3],y_temp_float_cur[5],y_temp_float_cur[7]]
        y_value_cur = [y_temp_float_cur[2],y_temp_float_cur[4],y_temp_float_cur[6],y_temp_float_cur[8]]

        #use current y as turth value
        x_bl_cur = (min(x_value_cur))#x bottom left
        y_bl_cur = (min(y_value_cur))#y bottom left
        x_tr_cur = (max(x_value_cur))#x top right
        y_tr_cur = (max(y_value_cur))#y top right

        bbox_curr = BoundingBox(x_bl_cur,y_bl_cur,x_tr_cur,y_tr_cur)
        #use current y as turth value
        x_bl_pre = (min(x_value_pre))#x bottom left
        y_bl_pre = (min(y_value_pre))#y bottom left
        x_tr_pre = (max(x_value_pre))#x top right
        y_tr_pre = (max(y_value_pre))#y top right

        bbox_prev = BoundingBox(x_bl_pre,y_bl_pre,x_tr_pre,y_tr_pre)

        target_pad, _, _, _ =  cropPadImage(bbox_prev, image_prev)
        curr_search_region, curr_search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_prev, image_curr)

        bbox_curr_gt = bbox_curr
        bbox_curr_gt_recentered = BoundingBox(0, 0, 0, 0)
        bbox_curr_gt_recentered = bbox_curr_gt.recenter(curr_search_location, edge_spacing_x, edge_spacing_y, bbox_curr_gt_recentered)
        bbox_curr_gt_recentered.scale(curr_search_region)

        curr_search_region = preprocess(curr_search_region)
        target_regoin = preprocess(target_pad)
        bbox_curr_y = [bbox_curr_gt_recentered.x1, bbox_curr_gt_recentered.y1, bbox_curr_gt_recentered.x2, bbox_curr_gt_recentered.y2]

        batch_x_prev.append(target_regoin)
        batch_x_curr.append(curr_search_region)
        batch_gt_prev.append(bbox_prev)
        batch_y_curr.append(bbox_curr_y)

        # image_test_cur = curr_search_region + np.array([104, 117, 123])
        # image_test_pre = target_regoin + np.array([104, 117, 123])
        # vis = np.concatenate((image_test_cur, image_test_pre), axis=1)
        # cv2.imwrite('./output/image_output/image_vot_cur_pre_'+format(file_num, '03d')+'_'+format(num_img, '03d')+'.jpg', vis)
#For vot video dataset load
#######################################################################
for file_num in range(fisrt_file_index_vot, end_file_index_vot):

    # text_file = open("./Track_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    text_file = open("./Track_human_dataset/vot_video000"+format(file_num, '02d')+"/groundtruth.txt", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    # text_file = open("./Track_position_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/

    groundtruth_frame = text_file.readlines()
    text_file.close()

    frame_len = len(groundtruth_frame)
    end_frame = frame_len/frame_step_vot-1-6
    print 'len(groundtruth_frame):', frame_len

    for num_img in range(6, frame_len, frame_step_vot):
        print 'num_img: ', num_img

        y_temp_cur = groundtruth_frame[num_img-1]
        y_temp_cur = y_temp_cur.split(",")
        y_temp_float_cur = np.array(y_temp_cur)
        y_temp_float_cur = y_temp_float_cur.astype(np.float)

        y_temp_pre = groundtruth_frame[num_img-1-frame_step_vot]
        y_temp_pre = y_temp_pre.split(",")
        y_temp_float_pre = np.array(y_temp_pre)
        y_temp_float_pre = y_temp_float_pre.astype(np.float)

        frame_index_cur = num_img
        frame_index_pre = num_img - frame_step_vot

        print 'file_num', file_num
        print 'frame_index_cur', frame_index_cur , ' frame_index_pre', frame_index_pre

        # image_path_cur = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_path_cur = './Track_human_dataset/vot_video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        # image_path_cur = './Track_position_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_curr = cv2.imread(image_path_cur)
        image_orginal_curr.append(image_read_cur)

        # image_path_pre = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_path_pre = './Track_human_dataset/vot_video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        # image_path_pre = './Track_position_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_prev = cv2.imread(image_path_pre)
        image_orginal_prev.append(image_read_pre)

        #Use previous frame to crop previous frame and current frame, and use current y as turth value
        x_value_pre = [y_temp_float_pre[0],y_temp_float_pre[2],y_temp_float_pre[4],y_temp_float_pre[6]]
        y_value_pre = [y_temp_float_pre[1],y_temp_float_pre[3],y_temp_float_pre[5],y_temp_float_pre[7]]
        x_value_cur = [y_temp_float_cur[0],y_temp_float_cur[2],y_temp_float_cur[4],y_temp_float_cur[6]]
        y_value_cur = [y_temp_float_cur[1],y_temp_float_cur[3],y_temp_float_cur[5],y_temp_float_cur[7]]

        #use current y as turth value
        x_bl_cur = (min(x_value_cur))#x bottom left
        y_bl_cur = (min(y_value_cur))#y bottom left
        x_tr_cur = (max(x_value_cur))#x top right
        y_tr_cur = (max(y_value_cur))#y top right

        bbox_curr = BoundingBox(x_bl_cur,y_bl_cur,x_tr_cur,y_tr_cur)
        #use current y as turth value
        x_bl_pre = (min(x_value_pre))#x bottom left
        y_bl_pre = (min(y_value_pre))#y bottom left
        x_tr_pre = (max(x_value_pre))#x top right
        y_tr_pre = (max(y_value_pre))#y top right

        bbox_prev = BoundingBox(x_bl_pre,y_bl_pre,x_tr_pre,y_tr_pre)

        target_pad, _, _, _ =  cropPadImage(bbox_prev, image_prev)
        curr_search_region, curr_search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_prev, image_curr)

        bbox_curr_gt = bbox_curr
        bbox_curr_gt_recentered = BoundingBox(0, 0, 0, 0)
        bbox_curr_gt_recentered = bbox_curr_gt.recenter(curr_search_location, edge_spacing_x, edge_spacing_y, bbox_curr_gt_recentered)
        bbox_curr_gt_recentered.scale(curr_search_region)

        curr_search_region = preprocess(curr_search_region)
        target_regoin = preprocess(target_pad)
        bbox_curr_y = [bbox_curr_gt_recentered.x1, bbox_curr_gt_recentered.y1, bbox_curr_gt_recentered.x2, bbox_curr_gt_recentered.y2]

        batch_x_prev.append(target_regoin)
        batch_x_curr.append(curr_search_region)
        batch_gt_prev.append(bbox_prev)
        batch_y_curr.append(bbox_curr_y)

        # image_test_cur = curr_search_region + np.array([104, 117, 123])
        # image_test_pre = target_regoin + np.array([104, 117, 123])
        # vis = np.concatenate((image_test_cur, image_test_pre), axis=1)
        # cv2.imwrite('./output/image_output/image_vot_cur_pre_'+format(file_num, '03d')+'_'+format(num_img, '03d')+'.jpg', vis)

print 'batch_x_curr length', len(batch_x_curr)
print 'batch_x_prev length', len(batch_x_prev)
print 'batch_y_curr length', len(batch_y_curr)
#######################################################################
print 'initail_frame: ', initail_frame
print 'end_frame: ', end_frame

vide_output = []

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Restore variables from disk.
    saver.restore(sess, "./tmp/"+model_name)
    print("\nModel restored.")

    for batch_index in range(len(batch_x_prev)):
        if batch_index == 0:
            bbox_prev_tight = batch_gt_prev[batch_index]
            bbox_curr_prior_tight = batch_gt_prev[batch_index]

        image_prev_input = image_orginal_prev[batch_index]
        image_curr_input = image_orginal_curr[batch_index]

        target_pad, _, _,  _ = cropPadImage(bbox_prev_tight, image_prev_input)
        cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_prior_tight, image_curr_input)

        curr_search_region = preprocess(cur_search_region)
        target_regoin = preprocess(target_pad)

        bbox_estimate = sess.run(logits, feed_dict={
                    x_cur: [curr_search_region],
                    x_pre: [target_regoin],
                    y: [[1,1,1,1]],
                    keep_prob: 1})

        bbox_estimate = BoundingBox(bbox_estimate[0, 0], bbox_estimate[0, 1], bbox_estimate[0, 2], bbox_estimate[0, 3])

        # Inplace correction of bounding box
        bbox_estimate.unscale(curr_search_region)

        bbox_output_unscale = [bbox_estimate.x1, bbox_estimate.y1, bbox_estimate.x2, bbox_estimate.y2]
        bbox_output_unscale = [int(i) for i in bbox_output_unscale]

        img_out_curr_unscale = cv2.rectangle(curr_search_region,(bbox_output_unscale[0],bbox_output_unscale[1]),(bbox_output_unscale[2],bbox_output_unscale[3]),(0,255,0),1)
        img_out_curr_unscale = img_out_curr_unscale + np.array([104, 117, 123])
        img_out_prev_unscale = target_regoin + np.array([104, 117, 123])
        vis = np.concatenate((img_out_curr_unscale, img_out_prev_unscale), axis=1)
        cv2.imwrite('./output/image_output/image_unscale_pre_cur'+format(batch_index, '03d')+'.jpg', vis)

        bbox_estimate.uncenter(image_curr_input, search_location, edge_spacing_x, edge_spacing_y)

        image_prev_input = image_curr_input
        bbox_prev_tight = bbox_estimate
        bbox_curr_prior_tight = bbox_estimate

        bbox_output = [bbox_estimate.x1, bbox_estimate.y1, bbox_estimate.x2, bbox_estimate.y2]
        bbox_output = [int(i) for i in bbox_output]
        print 'bbox_output int', bbox_output

        img_out_curr = cv2.rectangle(image_curr_input,(bbox_output[0],bbox_output[1]),(bbox_output[2],bbox_output[3]),(0,255,0),1)
        img_out_curr = img_out_curr# + np.array([104, 117, 123])
        cv2.imwrite('./output/image_output/image_croping_pre_cur'+format(batch_index, '03d')+'.jpg', img_out_curr)

        outputdata = img_out_curr.astype(np.uint8)
        vide_output.append(outputdata)

skvideo.io.vwrite("./output/video_output/track_video_goturn_est"+format(fisrt_file_index, '02d')+".mp4", vide_output)
print "video built"

print "\nmodel_name",model_name
