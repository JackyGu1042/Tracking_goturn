import cv2
import os
import glob
import numpy as np
from time import gmtime, strftime
import skvideo.io
import math

from load_alov import video
from load_alov import frame
from load_alov import BoundingBox
from load_alov import annotation
from load_alov import cropPadImage
from load_alov import preprocess
from load_alov import load_annotation_file

width_resize = 227
height_resize = 227
channel_resize = 3

batch_size = 50

lamda_shift = 5
lamda_scale = 15
min_scale = 0.4
max_scale = 0.4

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

videos = []
frame_total = 0

num_annotations = 0
list_of_annotations_out = []
################################## Load .ann gt file in ALOV ########################################
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

######### Ramdom select image to train ################
images = list_of_annotations_out

batch_x_prev =[]
batch_x_curr = []
batch_y_curr = []

while len(batch_x_prev) < batch_size:
    print '\ni', i
    curr_image_index = np.random.randint(0, len(images))
    list_annotations = images[curr_image_index]
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


# batch_image_prev =[]
# batch_image_curr = []
# batch_gt_curr = []
#
# while len(batch_image_prev) < batch_size:
#     video_num = np.random.randint(0,len(videos))
#     video = videos[video_num]
#     annotations = video.annotations
#
#     if len(annotations) < 2:
#         print 'Error - video {} has only {} annotations', video.video_path, len(annotations)
#
#     ann_index = np.random.randint(0, len(annotations)-2)
#     print 'ann_index', ann_index
#
#     frame_num_prev, image_prev, bbox_prev = video.load_annotation(ann_index)
#     frame_num_curr, image_curr, bbox_curr = video.load_annotation(ann_index + 1)
#
#     ################# preprocess the input image ############################
#     target_pad, _, _, _ =  cropPadImage(bbox_prev, image_prev)
#     curr_search_region, curr_search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_prev, image_curr)
#
#     bbox_curr_gt = bbox_curr
#     bbox_curr_gt_recentered = BoundingBox(0, 0, 0, 0)
#     bbox_curr_gt_recentered = bbox_curr_gt.recenter(curr_search_location, edge_spacing_x, edge_spacing_y, bbox_curr_gt_recentered)
#
#     curr_search_region_box = curr_search_region
#     cv2.rectangle(curr_search_region_box,(int(bbox_curr_gt_recentered.x1),int(bbox_curr_gt_recentered.y1)),(int(bbox_curr_gt_recentered.x2),int(bbox_curr_gt_recentered.y2)),(0,200,0),1)
#
#     vis = np.concatenate((target_pad, curr_search_region_box), axis=1)
#     cv2.imwrite('./image_result/image_orignal_pre_cur'+format(len(batch_image_prev), '03d')+'.jpg', vis)
#
#     bbox_curr_gt_recentered.scale(curr_search_region)
#
#     curr_search_region = preprocess(curr_search_region)
#     target_regoin = preprocess(target_pad)
#
#     batch_image_prev.append(target_regoin)
#     batch_image_curr.append(curr_search_region)
#     batch_gt_curr.append(bbox_curr_gt_recentered)
#
# print 'batch-size', batch_size
# print 'len(batch_image_prev)',len(batch_image_prev)







# class frame:
#     def __init__(self,frame_num,bbox):
#         self.frame_num = frame_num
#         self.bbox = bbox
#
# class video:
#     def __init__(self,video_path):
#         self.video_path = video_path
#         self.all_frames = []
#         self.annotations = []
#
#     def load_annotation(self, annotation_index):
#         ann_frame = self.annotations[annotation_index]
#         frame_num = ann_frame.frame_num
#         bbox = ann_frame.bbox
#
#         video_path = self.video_path
#         image_files =  self.all_frames
#
#         assert(len(image_files) > 0)
#         assert(frame_num < len(image_files))
#
#         image = cv2.imread(image_files[frame_num])
#         return frame_num, image, bbox
#
# class BoundingBox:
#     def __init__(self, x1, y1, x2, y2):
#         """bounding box """
#
#         self.x1 = x1
#         self.y1 = y1
#         self.x2 = x2
#         self.y2 = y2
#         self.frame_num = 0
#         self.kContextFactor = 2
#         self.kScaleFactor = 10
#
#     def get_center_x(self):
#         """TODO: Docstring for get_center_x.
#         :returns: TODO
#
#         """
#         return (self.x1 + self.x2)/2.
#
#     def get_center_y(self):
#         """TODO: Docstring for get_center_y.
#         :returns: TODO
#
#         """
#         return (self.y1 + self.y2)/2.
#
#     def compute_output_height(self):
#         """TODO: Docstring for compute_output_height.
#         :returns: TODO
#
#         """
#         bbox_height = self.y2 - self.y1
#         output_height = self.kContextFactor * bbox_height
#
#         return max(1.0, output_height)
#
#     def compute_output_width(self):
#         """TODO: Docstring for compute_output_width.
#         :returns: TODO
#
#         """
#         bbox_width = self.x2 - self.x1
#         output_width = self.kContextFactor * bbox_width
#
#         return max(1.0, output_width)
#
#     def edge_spacing_x(self):
#         """TODO: Docstring for edge_spacing_x.
#         :returns: TODO
#
#         """
#         output_width = self.compute_output_width()
#         bbox_center_x = self.get_center_x()
#
#         return max(0.0, (output_width / 2) - bbox_center_x)
#
#     def edge_spacing_y(self):
#         """TODO: Docstring for edge_spacing_y.
#         :returns: TODO
#
#         """
#         output_height = self.compute_output_height()
#         bbox_center_y = self.get_center_y()
#
#         return max(0.0, (output_height / 2) - bbox_center_y)
#
#     def unscale(self, image):
#         """TODO: Docstring for unscale.
#         :returns: TODO
#
#         """
#         height = image.shape[0]
#         width = image.shape[1]
#
#         self.x1 = self.x1 / self.kScaleFactor
#         self.x2 = self.x2 / self.kScaleFactor
#         self.y1 = self.y1 / self.kScaleFactor
#         self.y2 = self.y2 / self.kScaleFactor
#
#         self.x1 = self.x1 * width
#         self.x2 = self.x2 * width
#         self.y1 = self.y1 * height
#         self.y2 = self.y2 * height
#
#     def uncenter(self, raw_image, search_location, edge_spacing_x, edge_spacing_y):
#         """TODO: Docstring for uncenter.
#         :returns: TODO
#
#         """
#         self.x1 = max(0.0, self.x1 + search_location.x1 - edge_spacing_x)
#         self.y1 = max(0.0, self.y1 + search_location.y1 - edge_spacing_y)
#         self.x2 = min(raw_image.shape[1], self.x2 + search_location.x1 - edge_spacing_x)
#         self.y2 = min(raw_image.shape[0], self.y2 + search_location.y1 - edge_spacing_y)
#
#     def recenter(self, search_loc, edge_spacing_x, edge_spacing_y, bbox_gt_recentered):
#         """TODO: Docstring for recenter.
#         :returns: TODO
#
#         """
#         bbox_gt_recentered.x1 = self.x1 - search_loc.x1 + edge_spacing_x
#         bbox_gt_recentered.y1 = self.y1 - search_loc.y1 + edge_spacing_y
#         bbox_gt_recentered.x2 = self.x2 - search_loc.x1 + edge_spacing_x
#         bbox_gt_recentered.y2 = self.y2 - search_loc.y1 + edge_spacing_y
#
#         return bbox_gt_recentered
#
#     def scale(self, image):
#         """TODO: Docstring for scale.
#         :returns: TODO
#
#         """
#         height = image.shape[0]
#         width = image.shape[1]
#
#         self.x1 = self.x1 / width
#         self.y1 = self.y1 / height
#         self.x2 = self.x2 / width
#         self.y2 = self.y2 / height
#
#         self.x1 = self.x1 * self.kScaleFactor
#         self.y1 = self.y1 * self.kScaleFactor
#         self.x2 = self.x2 * self.kScaleFactor
#         self.y2 = self.y2 * self.kScaleFactor
#
# def cropPadImage(bbox_tight, image):
#     """TODO: Docstring for cropPadImage.
#     :returns: TODO
#
#     """
#     pad_image_location = computeCropPadImageLocation(bbox_tight, image)
#     roi_left = min(pad_image_location.x1, (image.shape[1] - 1))
#     roi_bottom = min(pad_image_location.y1, (image.shape[0] - 1))
#     roi_width = min(image.shape[1], max(1.0, math.ceil(pad_image_location.x2 - pad_image_location.x1)))
#     roi_height = min(image.shape[0], max(1.0, math.ceil(pad_image_location.y2 - pad_image_location.y1)))
#
#     err = 0.000000001 # To take care of floating point arithmetic errors
#     cropped_image = image[int(roi_bottom + err):int(roi_bottom + roi_height), int(roi_left + err):int(roi_left + roi_width)]
#
#     # Padded output width and height
#     output_width = max(math.ceil(bbox_tight.compute_output_width()), roi_width)
#     output_height = max(math.ceil(bbox_tight.compute_output_height()), roi_height)
#
#     if image.ndim > 2:
#         output_image = np.zeros((int(output_height), int(output_width), image.shape[2]), dtype=image.dtype)
#     else:
#         output_image = np.zeros((int(output_height), int(output_width)), dtype=image.dtype)
#
#     # Center of the bounding box
#     edge_spacing_x = min(bbox_tight.edge_spacing_x(), (image.shape[1] - 1))
#     edge_spacing_y = min(bbox_tight.edge_spacing_y(), (image.shape[0] - 1))
#
#     # rounding should be done to match the width and height
#     output_image[int(edge_spacing_y):int(edge_spacing_y) + cropped_image.shape[0], int(edge_spacing_x):int(edge_spacing_x) + cropped_image.shape[1]] = cropped_image
#     return output_image, pad_image_location, edge_spacing_x, edge_spacing_y
#
#
# def computeCropPadImageLocation(bbox_tight, image):
#     """TODO: Docstring for computeCropPadImageLocation.
#     :returns: TODO
#
#     """
#     # Center of the bounding box
#     bbox_center_x = bbox_tight.get_center_x()
#     bbox_center_y = bbox_tight.get_center_y()
#
#     image_height = image.shape[0]
#     image_width = image.shape[1]
#
#     # Padded output width and height
#     output_width = bbox_tight.compute_output_width()
#     output_height = bbox_tight.compute_output_height()
#
#     roi_left = max(0.0, bbox_center_x - (output_width / 2.))
#     roi_bottom = max(0.0, bbox_center_y - (output_height / 2.))
#
#     # Padded roi width
#     left_half = min(output_width / 2., bbox_center_x)
#     right_half = min(output_width / 2., image_width - bbox_center_x)
#     roi_width = max(1.0, left_half + right_half)
#
#     # Padded roi height
#     top_half = min(output_height / 2., bbox_center_y)
#     bottom_half = min(output_height / 2., image_height - bbox_center_y)
#     roi_height = max(1.0, top_half + bottom_half)
#
#     # Padded image location in the original image
#     objPadImageLocation = BoundingBox(roi_left, roi_bottom, roi_left + roi_width, roi_bottom + roi_height)
#
#     return objPadImageLocation
#
# def preprocess(image):
#         """TODO: Docstring for preprocess.
#
#         :arg1: TODO
#         :returns: TODO
#
#         """
#         num_channels = channel_resize
#         if num_channels == 1 and image.shape[2] == 3:
#             image_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         elif num_channels == 1 and image.shape[2] == 4:
#             image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
#         elif num_channels == 3 and image.shape[2] == 4:
#             image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
#         elif num_channels == 3 and image.shape[2] == 2:
#             image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#         else:
#             image_out = image
#
#         if image_out.shape != (height_resize, width_resize, channel_resize):
#             image_out = cv2.resize(image_out, (width_resize, height_resize), interpolation=cv2.INTER_CUBIC)
#
#         image_out = np.float32(image_out)
#         image_out -= np.array([104, 117, 123])
#         # image_out = np.transpose(image_out, [2, 0, 1])
#         return image_out
#
#
# imagenet_folder_image = './ALOV/images'
# imagenet_folder_gt = './ALOV/gt'
#
# if not os.path.isdir(imagenet_folder_image):
#     print('{} is not a valid directory'.format(imagenet_folder_image))
# if not os.path.isdir(imagenet_folder_gt):
#     print('{} is not a valid directory'.format(imagenet_folder_gt))
#
# imagenet_subdirs_image = sorted([dir_name for dir_name in os.listdir(imagenet_folder_image) if os.path.isdir(os.path.join(imagenet_folder_image, dir_name))])
# # print imagenet_subdirs_image
#
# imagenet_subdirs_gt = sorted([dir_name for dir_name in os.listdir(imagenet_folder_gt) if os.path.isdir(os.path.join(imagenet_folder_gt, dir_name))])
# # print imagenet_subdirs_gt
# class frame:
#     def __init__(self,frame_num,bbox):
#         self.frame_num = frame_num
#         self.bbox = bbox
#
# class video:
#     def __init__(self,video_path):
#         self.video_path = video_path
#         self.all_frames = []
#         self.annotations = []
#
#     def load_annotation(self, annotation_index):
#         ann_frame = self.annotations[annotation_index]
#         frame_num = ann_frame.frame_num
#         bbox = ann_frame.bbox
#
#         video_path = self.video_path
#         image_files =  self.all_frames
#
#         assert(len(image_files) > 0)
#         assert(frame_num < len(image_files))
#
#         image = cv2.imread(image_files[frame_num])
#         return frame_num, image, bbox
#
# class BoundingBox:
#     def __init__(self, x1, y1, x2, y2):
#         """bounding box """
#
#         self.x1 = x1
#         self.y1 = y1
#         self.x2 = x2
#         self.y2 = y2
#         self.frame_num = 0
#         self.kContextFactor = 2
#         self.kScaleFactor = 10
#
#     def get_center_x(self):
#         """TODO: Docstring for get_center_x.
#         :returns: TODO
#
#         """
#         return (self.x1 + self.x2)/2.
#
#     def get_center_y(self):
#         """TODO: Docstring for get_center_y.
#         :returns: TODO
#
#         """
#         return (self.y1 + self.y2)/2.
#
#     def compute_output_height(self):
#         """TODO: Docstring for compute_output_height.
#         :returns: TODO
#
#         """
#         bbox_height = self.y2 - self.y1
#         output_height = self.kContextFactor * bbox_height
#
#         return max(1.0, output_height)
#
#     def compute_output_width(self):
#         """TODO: Docstring for compute_output_width.
#         :returns: TODO
#
#         """
#         bbox_width = self.x2 - self.x1
#         output_width = self.kContextFactor * bbox_width
#
#         return max(1.0, output_width)
#
#     def edge_spacing_x(self):
#         """TODO: Docstring for edge_spacing_x.
#         :returns: TODO
#
#         """
#         output_width = self.compute_output_width()
#         bbox_center_x = self.get_center_x()
#
#         return max(0.0, (output_width / 2) - bbox_center_x)
#
#     def edge_spacing_y(self):
#         """TODO: Docstring for edge_spacing_y.
#         :returns: TODO
#
#         """
#         output_height = self.compute_output_height()
#         bbox_center_y = self.get_center_y()
#
#         return max(0.0, (output_height / 2) - bbox_center_y)
#
#     def unscale(self, image):
#         """TODO: Docstring for unscale.
#         :returns: TODO
#
#         """
#         height = image.shape[0]
#         width = image.shape[1]
#
#         self.x1 = self.x1 / self.kScaleFactor
#         self.x2 = self.x2 / self.kScaleFactor
#         self.y1 = self.y1 / self.kScaleFactor
#         self.y2 = self.y2 / self.kScaleFactor
#
#         self.x1 = self.x1 * width
#         self.x2 = self.x2 * width
#         self.y1 = self.y1 * height
#         self.y2 = self.y2 * height
#
#     def uncenter(self, raw_image, search_location, edge_spacing_x, edge_spacing_y):
#         """TODO: Docstring for uncenter.
#         :returns: TODO
#
#         """
#         self.x1 = max(0.0, self.x1 + search_location.x1 - edge_spacing_x)
#         self.y1 = max(0.0, self.y1 + search_location.y1 - edge_spacing_y)
#         self.x2 = min(raw_image.shape[1], self.x2 + search_location.x1 - edge_spacing_x)
#         self.y2 = min(raw_image.shape[0], self.y2 + search_location.y1 - edge_spacing_y)
#
#     def recenter(self, search_loc, edge_spacing_x, edge_spacing_y, bbox_gt_recentered):
#         """TODO: Docstring for recenter.
#         :returns: TODO
#
#         """
#         bbox_gt_recentered.x1 = self.x1 - search_loc.x1 + edge_spacing_x
#         bbox_gt_recentered.y1 = self.y1 - search_loc.y1 + edge_spacing_y
#         bbox_gt_recentered.x2 = self.x2 - search_loc.x1 + edge_spacing_x
#         bbox_gt_recentered.y2 = self.y2 - search_loc.y1 + edge_spacing_y
#
#         return bbox_gt_recentered
#
#     def scale(self, image):
#         """TODO: Docstring for scale.
#         :returns: TODO
#
#         """
#         height = image.shape[0]
#         width = image.shape[1]
#
#         self.x1 = self.x1 / width
#         self.y1 = self.y1 / height
#         self.x2 = self.x2 / width
#         self.y2 = self.y2 / height
#
#         self.x1 = self.x1 * self.kScaleFactor
#         self.y1 = self.y1 * self.kScaleFactor
#         self.x2 = self.x2 * self.kScaleFactor
#         self.y2 = self.y2 * self.kScaleFactor
#
# def cropPadImage(bbox_tight, image):
#     """TODO: Docstring for cropPadImage.
#     :returns: TODO
#
#     """
#     pad_image_location = computeCropPadImageLocation(bbox_tight, image)
#     roi_left = min(pad_image_location.x1, (image.shape[1] - 1))
#     roi_bottom = min(pad_image_location.y1, (image.shape[0] - 1))
#     roi_width = min(image.shape[1], max(1.0, math.ceil(pad_image_location.x2 - pad_image_location.x1)))
#     roi_height = min(image.shape[0], max(1.0, math.ceil(pad_image_location.y2 - pad_image_location.y1)))
#
#     err = 0.000000001 # To take care of floating point arithmetic errors
#     cropped_image = image[int(roi_bottom + err):int(roi_bottom + roi_height), int(roi_left + err):int(roi_left + roi_width)]
#
#     # Padded output width and height
#     output_width = max(math.ceil(bbox_tight.compute_output_width()), roi_width)
#     output_height = max(math.ceil(bbox_tight.compute_output_height()), roi_height)
#
#     if image.ndim > 2:
#         output_image = np.zeros((int(output_height), int(output_width), image.shape[2]), dtype=image.dtype)
#     else:
#         output_image = np.zeros((int(output_height), int(output_width)), dtype=image.dtype)
#
#     # Center of the bounding box
#     edge_spacing_x = min(bbox_tight.edge_spacing_x(), (image.shape[1] - 1))
#     edge_spacing_y = min(bbox_tight.edge_spacing_y(), (image.shape[0] - 1))
#
#     # rounding should be done to match the width and height
#     output_image[int(edge_spacing_y):int(edge_spacing_y) + cropped_image.shape[0], int(edge_spacing_x):int(edge_spacing_x) + cropped_image.shape[1]] = cropped_image
#     return output_image, pad_image_location, edge_spacing_x, edge_spacing_y
#
#
# def computeCropPadImageLocation(bbox_tight, image):
#     """TODO: Docstring for computeCropPadImageLocation.
#     :returns: TODO
#
#     """
#     # Center of the bounding box
#     bbox_center_x = bbox_tight.get_center_x()
#     bbox_center_y = bbox_tight.get_center_y()
#
#     image_height = image.shape[0]
#     image_width = image.shape[1]
#
#     # Padded output width and height
#     output_width = bbox_tight.compute_output_width()
#     output_height = bbox_tight.compute_output_height()
#
#     roi_left = max(0.0, bbox_center_x - (output_width / 2.))
#     roi_bottom = max(0.0, bbox_center_y - (output_height / 2.))
#
#     # Padded roi width
#     left_half = min(output_width / 2., bbox_center_x)
#     right_half = min(output_width / 2., image_width - bbox_center_x)
#     roi_width = max(1.0, left_half + right_half)
#
#     # Padded roi height
#     top_half = min(output_height / 2., bbox_center_y)
#     bottom_half = min(output_height / 2., image_height - bbox_center_y)
#     roi_height = max(1.0, top_half + bottom_half)
#
#     # Padded image location in the original image
#     objPadImageLocation = BoundingBox(roi_left, roi_bottom, roi_left + roi_width, roi_bottom + roi_height)
#
#     return objPadImageLocation
#
# def preprocess(image):
#         """TODO: Docstring for preprocess.
#
#         :arg1: TODO
#         :returns: TODO
#
#         """
#         num_channels = channel_resize
#         if num_channels == 1 and image.shape[2] == 3:
#             image_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         elif num_channels == 1 and image.shape[2] == 4:
#             image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
#         elif num_channels == 3 and image.shape[2] == 4:
#             image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
#         elif num_channels == 3 and image.shape[2] == 2:
#             image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#         else:
#             image_out = image
#
#         if image_out.shape != (height_resize, width_resize, channel_resize):
#             image_out = cv2.resize(image_out, (width_resize, height_resize), interpolation=cv2.INTER_CUBIC)
#
#         image_out = np.float32(image_out)
#         image_out -= np.array([104, 117, 123])
#         # image_out = np.transpose(image_out, [2, 0, 1])
#         return image_out


# alov_folder_image = './ALOV/images'
# alov_folder_gt = './ALOV/gt'
#
# if not os.path.isdir(alov_folder_image):
#     print('{} is not a valid directory'.format(alov_folder_image))
# if not os.path.isdir(alov_folder_gt):
#     print('{} is not a valid directory'.format(alov_folder_gt))
#
# alov_subdirs_image = sorted([dir_name for dir_name in os.listdir(alov_folder_image) if os.path.isdir(os.path.join(alov_folder_image, dir_name))])
# # print alov_subdirs_image
#
# alov_subdirs_gt = sorted([dir_name for dir_name in os.listdir(alov_folder_gt) if os.path.isdir(os.path.join(alov_folder_gt, dir_name))])
# # print alov_subdirs_gt
#
# videos = []
# frame_total = 0
# ################################## Load .ann gt file in ALOV ########################################
# for i, alov_sub_folder_gt in enumerate(alov_subdirs_gt):
#     # print alov_folder_gt,'/',alov_sub_folder_gt
#     annotations_files = sorted(glob.glob(os.path.join(alov_folder_gt, alov_sub_folder_gt, '*.ann')))
#     # print i,' ', annotations_files
#
#     for ann in annotations_files:
#         video_path = os.path.join(alov_folder_image, alov_subdirs_image[i], ann.split('/')[-1].split('.')[0])
#         # print video_path
#         video_tmp = video(video_path)
#
#         all_frames = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
#         # print len(all_frames)
#         video_tmp.all_frames = all_frames
#
#         with open(ann, 'r') as f:
#             data = f.read().rstrip().split('\n')
#             for bb in data:
#                 frame_num, ax, ay, bx, by, cx, cy, dx, dy = bb.split()
#                 frame_num, ax, ay, bx, by, cx, cy, dx, dy = int(frame_num), float(ax), float(ay), float(bx), float(by), float(cx), float(cy), float(dx), float(dy)
#
#                 x1 = min(ax, min(bx, min(cx, dx))) - 1
#                 y1 = min(ay, min(by, min(cy, dy))) - 1
#                 x2 = max(ax, max(bx, max(cx, dx))) - 1
#                 y2 = max(ay, max(by, max(cy, dy))) - 1
#
#                 # print x1,y1,x2,y2
#                 bbox = BoundingBox(x1,y1,x2,y2)
#                 frame_tmp = frame(frame_num,bbox)
#
#                 video_tmp.annotations.append(frame_tmp)
#
#                 assert(len(all_frames) > 0)
#                 assert(frame_num-1 < len(all_frames))
#
#
#         frame_total += len(video_tmp.annotations)
#         videos.append(video_tmp)
#
# print 'video file number:', len(videos)
# print 'total image number:', frame_total
# ######### Ramdom select image to train ################
# batch_image_prev =[]
# batch_image_curr = []
# batch_gt_curr = []
#
# while len(batch_image_prev) < batch_size:
#     video_num = np.random.randint(0,len(videos))
#     video = videos[video_num]
#     annotations = video.annotations
#
#     if len(annotations) < 2:
#         print 'Error - video {} has only {} annotations', video.video_path, len(annotations)
#
#     ann_index = np.random.randint(0, len(annotations)-2)
#     print 'ann_index', ann_index
#
#     frame_num_prev, image_prev, bbox_prev = video.load_annotation(ann_index)
#     frame_num_curr, image_curr, bbox_curr = video.load_annotation(ann_index + 1)
#
#     ################# preprocess the input image ############################
#     target_pad, _, _, _ =  cropPadImage(bbox_prev, image_prev)
#     curr_search_region, curr_search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_prev, image_curr)
#
#     bbox_curr_gt = bbox_curr
#     bbox_curr_gt_recentered = BoundingBox(0, 0, 0, 0)
#     bbox_curr_gt_recentered = bbox_curr_gt.recenter(curr_search_location, edge_spacing_x, edge_spacing_y, bbox_curr_gt_recentered)
#
#     curr_search_region_box = curr_search_region
#     cv2.rectangle(curr_search_region_box,(int(bbox_curr_gt_recentered.x1),int(bbox_curr_gt_recentered.y1)),(int(bbox_curr_gt_recentered.x2),int(bbox_curr_gt_recentered.y2)),(0,200,0),1)
#
#     vis = np.concatenate((target_pad, curr_search_region_box), axis=1)
#     cv2.imwrite('./image_result/image_orignal_pre_cur'+format(len(batch_image_prev), '03d')+'.jpg', vis)
#
#     bbox_curr_gt_recentered.scale(curr_search_region)
#
#     curr_search_region = preprocess(curr_search_region)
#     target_regoin = preprocess(target_pad)
#
#     batch_image_prev.append(target_regoin)
#     batch_image_curr.append(curr_search_region)
#     batch_gt_curr.append(bbox_curr_gt_recentered)
#
# print 'batch-size', batch_size
# print 'len(batch_image_prev)',len(batch_image_prev)

# cv2.waitKey(0)

# print len(videos)
# print videos[0].video_path
# print videos[0].all_frames[0]
# print videos[0].annotations[0].frame_num
# print videos[0].annotations[0].bbox

# print frame_num_prev
# print bbox_prev
# cv2.imshow("prev image", image_prev)
#
# print frame_num_curr
# print bbox_curr
# cv2.imshow("curr image", image_curr)

# cv2.imshow("target_pad", target_pad)
# cv2.imshow("curr_search_region", curr_search_region)
# print 'curr_search_location', curr_search_location

# print 'video_num', video_num
# print 'annotations', annotations
