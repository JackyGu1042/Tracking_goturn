import os
import cv2
import math
import glob
import random
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
# from numpy import newaxis #for add one more dimension

# Parameters
learning_rate = 0.00001
kNumBatches = 5
batch_size = 50

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

lamda_shift = 5
lamda_scale = 15
min_scale = 0.4
max_scale = 0.4

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
    tf.summary.image('x_cur_pre', x_cur_pre, 30)

    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

with tf.name_scope('truth'):
    y = tf.placeholder(tf.float32, [None, n_output], name='y_input')

with tf.name_scope('network'):
    with tf.name_scope('logits'):
        # Model
        logits = conv_net(x_cur, x_pre, keep_prob)

    # Define loss and optimizer
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.square(logits-y))
        cost_L1 = tf.reduce_mean(tf.square(logits-y)/2)
        #Send to TensorBoard to display
        tf.summary.scalar('cost', cost)
        tf.summary.scalar('cost_L1', cost_L1)
        # tf.summary.histogram('histogram', cost_L1)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
            .minimize(cost)

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

#################################################################################
imagenet_folder_image = './ILSVRC2012/images'
imagenet_folder_gt = './ILSVRC2012/gt'

if not os.path.isdir(imagenet_folder_image):
    print('{} is not a valid directory'.format(imagenet_folder_image))
if not os.path.isdir(imagenet_folder_gt):
    print('{} is not a valid directory'.format(imagenet_folder_gt))

imagenet_subdirs_image = sorted([dir_name for dir_name in os.listdir(imagenet_folder_image) if os.path.isdir(os.path.join(imagenet_folder_image, dir_name))])
# print '\nimagenet_subdirs_image',imagenet_subdirs_image

imagenet_subdirs_gt = sorted([dir_name for dir_name in os.listdir(imagenet_folder_gt) if os.path.isdir(os.path.join(imagenet_folder_gt, dir_name))])
# print '\nimagenet_subdirs_gt',imagenet_subdirs_gt

num_annotations = 0
list_of_annotations_out = []

################################## Load .ann gt file in imagenet ########################################
for i, imagenet_sub_folder_gt in enumerate(imagenet_subdirs_gt):
    # print imagenet_subdirs_gt,'/',imagenet_sub_folder_gt
    annotations_files = sorted(glob.glob(os.path.join(imagenet_folder_gt, imagenet_sub_folder_gt, '*.xml')))
    # print i,' ', annotations_files

    for ann in annotations_files:
        list_of_annotations, num_ann_curr = load_annotation_file(ann)
        num_annotations = num_annotations + num_ann_curr
        if len(list_of_annotations) == 0:
            continue
        list_of_annotations_out.append(list_of_annotations)

print 'len(list_of_annotations_out)', len(list_of_annotations_out)
print 'num_annotations', num_annotations

images_imagenet = list_of_annotations_out

#################################################################################
alov_folder_image = './ALOV/images'
alov_folder_gt = './ALOV/gt'

if not os.path.isdir(alov_folder_image):
    print('{} is not a valid directory'.format(alov_folder_image))
if not os.path.isdir(alov_folder_gt):
    print('{} is not a valid directory'.format(alov_folder_gt))

alov_subdirs_image = sorted([dir_name for dir_name in os.listdir(alov_folder_image) if os.path.isdir(os.path.join(alov_folder_image, dir_name))])
# print alov_subdirs_image

alov_subdirs_gt = sorted([dir_name for dir_name in os.listdir(alov_folder_gt) if os.path.isdir(os.path.join(alov_folder_gt, dir_name))])
# print alov_subdirs_gt

videos = []
frame_total = 0

################################## Load .ann gt file in ALOV ########################################
for i, alov_sub_folder_gt in enumerate(alov_subdirs_gt):
    # print alov_folder_gt,'/',alov_sub_folder_gt
    annotations_files = sorted(glob.glob(os.path.join(alov_folder_gt, alov_sub_folder_gt, '*.ann')))
    # print i,' ', annotations_files

    for ann in annotations_files:
        video_path = os.path.join(alov_folder_image, alov_subdirs_image[i], ann.split('/')[-1].split('.')[0])
        # print video_path
        video_tmp = video(video_path)

        all_frames = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
        # print len(all_frames)
        video_tmp.all_frames = all_frames

        with open(ann, 'r') as f:
            data = f.read().rstrip().split('\n')
            for bb in data:
                frame_num, ax, ay, bx, by, cx, cy, dx, dy = bb.split()
                frame_num, ax, ay, bx, by, cx, cy, dx, dy = int(frame_num), float(ax), float(ay), float(bx), float(by), float(cx), float(cy), float(dx), float(dy)

                x1 = min(ax, min(bx, min(cx, dx))) - 1
                y1 = min(ay, min(by, min(cy, dy))) - 1
                x2 = max(ax, max(bx, max(cx, dx))) - 1
                y2 = max(ay, max(by, max(cy, dy))) - 1

                # print x1,y1,x2,y2
                bbox = BoundingBox(x1,y1,x2,y2)
                frame_tmp = frame(frame_num,bbox)

                video_tmp.annotations.append(frame_tmp)

                assert(len(all_frames) > 0)
                assert(frame_num-1 < len(all_frames))


        frame_total += len(video_tmp.annotations)
        videos.append(video_tmp)

print '\nvideo file number:', len(videos)
print 'total image number:', frame_total, '\n'

#######################################################

num_batches = 0
# Launch the graph
with tf.Session() as sess:
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    merged = tf.summary.merge_all() # tensorflow >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)

    sess.run(init)

    while num_batches < kNumBatches:
        batch_x_prev =[]
        batch_x_curr = []
        batch_y_curr = []

        while len(batch_x_prev) < batch_size:
            print '\ni', i
            curr_image_index = np.random.randint(0, len(images_imagenet))
            list_annotations = images_imagenet[curr_image_index]
            print 'curr_image_index', curr_image_index

            curr_ann_index = np.random.randint(0, len(list_annotations))
            random_ann = list_annotations[curr_ann_index]
            print 'curr_ann_index', curr_ann_index

            print 'image_path',imagenet_folder_image, random_ann.image_path, '.JPEG'
            print 'len(random_ann.image_path)',len(random_ann.image_path)

            if len(random_ann.image_path) < 15:
                print '********path mistake********'
                continue

            img_path = os.path.join(imagenet_folder_image, random_ann.image_path + '.JPEG')

            image = cv2.imread(img_path)

            img_height = image.shape[0]
            img_width = image.shape[1]

            sc_factor_1 = 1.0
            if img_height != random_ann.disp_height or img_width != random_ann.disp_width:
                sc_factor_1 = (img_height * 1.) / random_ann.disp_height
                sc_factor_2 = (img_width * 1.) / random_ann.disp_width

            bbox = random_ann.bbox
            bbox.x1 = bbox.x1 * sc_factor_1
            bbox.x2 = bbox.x2 * sc_factor_1
            bbox.y1 = bbox.y1 * sc_factor_1
            bbox.y2 = bbox.y2 * sc_factor_1

            bbox_prev = bbox
            image_prev = image

            # cv2.imshow('random image'+ format(i, '02d'), image)

            target_pad, _, _, _ =  cropPadImage(bbox_prev, image_prev)

            # cv2.imshow('prev croped image'+ format(i, '02d'), target_pad)

            bbox_curr_gt = bbox_prev
            bbox_curr_shift = BoundingBox(0, 0, 0, 0)
            bbox_curr_shift = bbox_curr_gt.shift(image_prev, lamda_scale, lamda_shift, min_scale, max_scale, True, bbox_curr_shift)
            rand_search_region, rand_search_location , edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_shift, image_prev)

            bbox_curr_gt = bbox_prev
            bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
            bbox_gt_recentered = bbox_curr_gt.recenter(rand_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered)

            # cv2.rectangle(rand_search_region,(int(bbox_gt_recentered.x1),int(bbox_gt_recentered.y1)),(int(bbox_gt_recentered.x2),int(bbox_gt_recentered.y2)),(0,200,0),1)
            # cv2.imwrite('./image_result/image_orignal_pre_cur'+format(len(batch_x_prev), '03d')+'.jpg', rand_search_region)

            bbox_gt_recentered.scale(rand_search_region)
            bbox_gt_scaled = bbox_gt_recentered

            rand_search_region = preprocess(rand_search_region)
            target_regoin = preprocess(target_pad)
            bbox_curr_y = [bbox_gt_scaled.x1, bbox_gt_scaled.y1, bbox_gt_scaled.x2, bbox_gt_scaled.y2]

            batch_x_prev.append(target_regoin)
            batch_x_curr.append(rand_search_region)
            batch_y_curr.append(bbox_curr_y)

        while len(batch_x_prev) < batch_size*2:
            video_num = np.random.randint(0,len(videos))
            video = videos[video_num]
            annotations = video.annotations

            if len(annotations) < 2:
                print 'Error - video {} has only {} annotations', video.video_path, len(annotations)

            ann_index = np.random.randint(0, len(annotations)-2)

            frame_num_prev, image_prev, bbox_prev = video.load_annotation(ann_index)
            frame_num_curr, image_curr, bbox_curr = video.load_annotation(ann_index + 1)

            ################# preprocess the input image ############################
            target_pad, _, _, _ =  cropPadImage(bbox_prev, image_prev)
            curr_search_region, curr_search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_prev, image_curr)

            bbox_curr_gt = bbox_curr
            bbox_curr_gt_recentered = BoundingBox(0, 0, 0, 0)
            bbox_curr_gt_recentered = bbox_curr_gt.recenter(curr_search_location, edge_spacing_x, edge_spacing_y, bbox_curr_gt_recentered)
            bbox_curr_gt_recentered.scale(curr_search_region)

            # curr_search_region_box = curr_search_region
            # cv2.rectangle(curr_search_region_box,(int(bbox_curr_gt_recentered.x1),int(bbox_curr_gt_recentered.y1)),(int(bbox_curr_gt_recentered.x2),int(bbox_curr_gt_recentered.y2)),(0,200,0),1)
            # vis = np.concatenate((target_pad, curr_search_region_box), axis=1)
            # cv2.imwrite('./image_result/image_orignal_pre_cur'+format(len(batch_x_prev), '03d')+'.jpg', vis)

            curr_search_region = preprocess(curr_search_region)
            target_regoin = preprocess(target_pad)
            bbox_curr_y = [bbox_curr_gt_recentered.x1, bbox_curr_gt_recentered.y1, bbox_curr_gt_recentered.x2, bbox_curr_gt_recentered.y2]

            batch_x_prev.append(target_regoin)
            batch_x_curr.append(curr_search_region)
            batch_y_curr.append(bbox_curr_y)

        print '\nnum-batches', num_batches

        logits_out = sess.run(logits, feed_dict={
            x_cur: batch_x_curr,
            x_pre: batch_x_prev,
            y: batch_y_curr,
            keep_prob: dropout})

        sess.run(optimizer, feed_dict={
            x_cur: batch_x_curr,
            x_pre: batch_x_prev,
            y: batch_y_curr,
            keep_prob: dropout})

        # Calculate batch loss and accuracy
        loss = sess.run(cost, feed_dict={
            x_cur: batch_x_curr,
            x_pre: batch_x_prev,
            y: batch_y_curr,
            keep_prob: 1.})

        # Calculate batch loss and accuracy
        loss_L1 = sess.run(cost_L1, feed_dict={
            x_cur: batch_x_curr,
            x_pre: batch_x_prev,
            y: batch_y_curr,
            keep_prob: 1.})


        rs = sess.run(merged, feed_dict={
            x_cur: batch_x_curr,
            x_pre: batch_x_prev,
            y: batch_y_curr,
            keep_prob: dropout})
        writer.add_summary(rs, num_batches)

        print('\nALOV iteration {:>2}, Batch {:>3} -'
              'loss_L1: {:>10.4f}'.format(
            num_batches + 1,
            batch_size,
            loss_L1))

        print 'logits_out[0]',logits_out[0]
        print "batch_y[0]", batch_y_curr[0]
        print 'learning_rate:', learning_rate , ' dropout: ', dropout

        num_batches = num_batches + 1

    time = strftime("%Y-%m-%d%H%M", gmtime())
    save_path = saver.save(sess, "./tmp/model64_4l_"+time+".ckpt") #
    print("Model saved in file: %s" % save_path)
