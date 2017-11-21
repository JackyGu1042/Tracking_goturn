import os
import cv2
import math
import glob
import caffe
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
from load_alov import preprocess_caffe
from load_alov import load_annotation_file

model_name = 'model64_4l_2017-11-190954_re2.ckpt'

# Parameters
file_num = 'car'

frame_step = 1
initail_frame = 2#frame_step + 1
end_frame = 251

# batch_gt_prev_initial = BoundingBox(411,180,484,211) #drunk
# batch_gt_prev_initial = BoundingBox(136,37,190,114) #diviad
batch_gt_prev_initial = BoundingBox(8,164,50,192) #car
# batch_gt_prev_initial = BoundingBox(343,175,352,191) #bolt
# batch_gt_prev_initial = BoundingBox(151,90,175,146) #bicycle
# batch_gt_prev_initial = BoundingBox(188,209,234,317) #basketball

n_output = 4

#Training resize image size
width_resize = 227
height_resize = 227
channel_resize = 3

deploy_proto = './net/tracker.prototxt'
caffe_model = './net/tracker.caffemodel'
gpu_id = 0

caffe.set_mode_gpu()
caffe.set_device(int(gpu_id))
# caffe.set_mode_cpu()

print 'Setting phase to test'
net = caffe.Net(deploy_proto, caffe_model, caffe.TEST)

net_num_inputs = net.blobs['image'].data[...].shape[0]
net_channels = net.blobs['image'].data[...].shape[1]
net_height = net.blobs['image'].data[...].shape[2]
net_width = net.blobs['image'].data[...].shape[3]

if net_num_inputs != 1:
    print 'Network should take exactly one input'

if net_channels != 1 and net_channels != 3:
    print 'Network should have 1 or 3 channels'

#######################################################################
#add one more dimension
image_orginal_cur = []
image_orginal_pre = []

image_path_cur = './Track_dataset/'+ file_num +'/00000' + format(initail_frame, '03d') + '.jpg'
# image_path_cur = './Track_human_dataset/video000'+format(file_num, '02d')+'/00000' + format(initail_frame, '03d') + '.jpg'
# image_path_cur = './Track_test/test_video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
image_read_cur = cv2.imread(image_path_cur)


#Stroe all the tracking frame original image data
for num_img in range(initail_frame, end_frame , frame_step):

    frame_index_cur = num_img
    frame_index_pre = num_img-frame_step

    print 'file_num', file_num
    print 'frame_index_cur', frame_index_cur , ' frame_index_pre', frame_index_pre

    image_path_cur = './Track_dataset/' + file_num +'/00000' + format(frame_index_cur, '03d') + '.jpg'
    # image_path_cur = './Track_human_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
    # image_path_cur = './Track_test/test_video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
    image_read_cur = cv2.imread(image_path_cur)
    image_orginal_cur.append(image_read_cur)

    image_path_pre = './Track_dataset/'+ file_num +'/00000' + format(frame_index_pre, '03d') + '.jpg'
    # image_path_pre = './Track_human_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
    # image_path_pre = './Track_test/test_video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
    image_read_pre = cv2.imread(image_path_pre)
    image_orginal_pre.append(image_read_pre)

    height_original, width_original = image_read_cur.shape[:2]
    print 'width_original', width_original, ' height_original', height_original



# bbox_output = [batch_gt_prev_initial.x1, batch_gt_prev_initial.y1, batch_gt_prev_initial.x2, batch_gt_prev_initial.y2]
# bbox_output = [int(i) for i in bbox_output]
# print 'bbox_output int', bbox_output
# image_initial = image_orginal_pre[0]
# img_out_initial = cv2.rectangle(image_initial,(bbox_output[0],bbox_output[1]),(bbox_output[2],bbox_output[3]),(0,255,0),1)
# cv2.imwrite('./output/image_output/initial_image.jpg', img_out_initial)


print 'image_read_cur length', len(image_read_cur)

#######################################################################

network_out = []
network_out_width_crop = []
network_out_height_crop = []
network_out_xstart_crop = []
network_out_ystart_crop = []
vide_output = []

print 'initail_frame: ', initail_frame
print 'end_frame: ', end_frame

for num_img in range(0,end_frame-initail_frame):
    if num_img == 0:
        bbox_prev_tight = batch_gt_prev_initial
        bbox_curr_prior_tight = batch_gt_prev_initial

    image_curr_input = image_orginal_cur[num_img]
    image_prev_input = image_orginal_pre[num_img]

    target_pad, _, _,  _ = cropPadImage(bbox_prev_tight, image_prev_input)
    cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_prior_tight, image_curr_input)

    # reshape the inputs

    net.blobs['image'].data.reshape(1, net_channels, net_height, net_width)
    net.blobs['target'].data.reshape(1, net_channels, net_height, net_width)
    net.blobs['bbox'].data.reshape(1, 4, 1, 1)

    curr_search_region = preprocess_caffe(cur_search_region)
    target_region = preprocess_caffe(target_pad)

    net.blobs['image'].data[...] = curr_search_region
    net.blobs['target'].data[...] = target_region
    net.forward()
    bbox_estimate = net.blobs['fc8'].data

    bbox_estimate = BoundingBox(bbox_estimate[0, 0], bbox_estimate[0, 1], bbox_estimate[0, 2], bbox_estimate[0, 3])

    # Inplace correction of bounding box
    print 'height', cur_search_region.shape[0], 'width', cur_search_region.shape[1]
    bbox_estimate.unscale(cur_search_region)

    bbox_output_unscale = [bbox_estimate.x1, bbox_estimate.y1, bbox_estimate.x2, bbox_estimate.y2]
    bbox_output_unscale = [int(i) for i in bbox_output_unscale]
    img_out_curr_unscale = cv2.rectangle(cur_search_region,(bbox_output_unscale[0],bbox_output_unscale[1]),(bbox_output_unscale[2],bbox_output_unscale[3]),(0,255,0),1)
    img_out_curr_unscale = img_out_curr_unscale
    img_out_prev_unscale = target_pad
    vis = np.concatenate((img_out_curr_unscale, img_out_prev_unscale), axis=1)
    cv2.imwrite('./output/image_output/image_unscale_pre_cur'+format(num_img, '03d')+'.jpg', vis)

    print 'search_location.x1', search_location.x1, 'search_location.y1', search_location.y1
    print 'edge_spacing_x',edge_spacing_x,' edge_spacing_y',edge_spacing_y
    bbox_estimate.uncenter(image_curr_input, search_location, edge_spacing_x, edge_spacing_y)

    image_prev_input = image_curr_input
    bbox_prev_tight = bbox_estimate
    bbox_curr_prior_tight = bbox_estimate

    bbox_output = [bbox_estimate.x1, bbox_estimate.y1, bbox_estimate.x2, bbox_estimate.y2]
    bbox_output = [int(i) for i in bbox_output]
    print 'bbox_output int', bbox_output

    img_out_curr = cv2.rectangle(image_curr_input,(bbox_output[0],bbox_output[1]),(bbox_output[2],bbox_output[3]),(0,255,0),1)
    img_out_curr = img_out_curr# + np.array([104, 117, 123])
    cv2.imwrite('./output/image_output/image_croping_pre_cur'+format(num_img, '03d')+'.jpg', img_out_curr)

    outputdata = img_out_curr.astype(np.uint8)
    vide_output.append(outputdata)

skvideo.io.vwrite("./output/video_output/track_video_goturn_est_"+ file_num +".mp4", vide_output)
print "video built"

print "\nmodel_name",model_name
