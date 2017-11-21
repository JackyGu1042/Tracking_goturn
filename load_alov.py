import cv2
import os
import glob
import random
import numpy as np
import math
import xml.etree.ElementTree as ET

width_resize = 227
height_resize = 227
channel_resize = 3

batch_size = 50

kMaxRatio = 0.66

RAND_MAX = 2147483647

class frame:
    def __init__(self,frame_num,bbox):
        self.frame_num = frame_num
        self.bbox = bbox


class video:
    def __init__(self,video_path):
        self.video_path = video_path
        self.all_frames = []
        self.annotations = []

    def load_annotation(self, annotation_index):
        ann_frame = self.annotations[annotation_index]
        frame_num = ann_frame.frame_num
        bbox = ann_frame.bbox

        video_path = self.video_path
        image_files =  self.all_frames

        assert(len(image_files) > 0)
        assert(frame_num < len(image_files))

        image = cv2.imread(image_files[frame_num])
        return frame_num, image, bbox


class BoundingBox:
    def __init__(self, x1, y1, x2, y2):
        """bounding box """

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame_num = 0
        self.kContextFactor = 2
        self.kScaleFactor = 10

    def get_center_x(self):
        """TODO: Docstring for get_center_x.
        :returns: TODO

        """
        return (self.x1 + self.x2)/2.

    def get_center_y(self):
        """TODO: Docstring for get_center_y.
        :returns: TODO

        """
        return (self.y1 + self.y2)/2.

    def compute_output_height(self):
        """TODO: Docstring for compute_output_height.
        :returns: TODO

        """
        bbox_height = self.y2 - self.y1
        output_height = self.kContextFactor * bbox_height

        return max(1.0, output_height)

    def compute_output_width(self):
        """TODO: Docstring for compute_output_width.
        :returns: TODO

        """
        bbox_width = self.x2 - self.x1
        output_width = self.kContextFactor * bbox_width

        return max(1.0, output_width)

    def edge_spacing_x(self):
        """TODO: Docstring for edge_spacing_x.
        :returns: TODO

        """
        output_width = self.compute_output_width()
        bbox_center_x = self.get_center_x()

        return max(0.0, (output_width / 2) - bbox_center_x)

    def edge_spacing_y(self):
        """TODO: Docstring for edge_spacing_y.
        :returns: TODO

        """
        output_height = self.compute_output_height()
        bbox_center_y = self.get_center_y()

        return max(0.0, (output_height / 2) - bbox_center_y)

    def unscale(self, image):
        """TODO: Docstring for unscale.
        :returns: TODO

        """
        height = image.shape[0]
        width = image.shape[1]

        self.x1 = self.x1 / self.kScaleFactor
        self.x2 = self.x2 / self.kScaleFactor
        self.y1 = self.y1 / self.kScaleFactor
        self.y2 = self.y2 / self.kScaleFactor

        self.x1 = self.x1 * width
        self.x2 = self.x2 * width
        self.y1 = self.y1 * height
        self.y2 = self.y2 * height

    def uncenter(self, raw_image, search_location, edge_spacing_x, edge_spacing_y):
        """TODO: Docstring for uncenter.
        :returns: TODO

        """
        self.x1 = max(0.0, self.x1 + search_location.x1 - edge_spacing_x)
        self.y1 = max(0.0, self.y1 + search_location.y1 - edge_spacing_y)
        self.x2 = min(raw_image.shape[1], self.x2 + search_location.x1 - edge_spacing_x)
        self.y2 = min(raw_image.shape[0], self.y2 + search_location.y1 - edge_spacing_y)

    def recenter(self, search_loc, edge_spacing_x, edge_spacing_y, bbox_gt_recentered):
        """TODO: Docstring for recenter.
        :returns: TODO

        """
        bbox_gt_recentered.x1 = self.x1 - search_loc.x1 + edge_spacing_x
        bbox_gt_recentered.y1 = self.y1 - search_loc.y1 + edge_spacing_y
        bbox_gt_recentered.x2 = self.x2 - search_loc.x1 + edge_spacing_x
        bbox_gt_recentered.y2 = self.y2 - search_loc.y1 + edge_spacing_y

        return bbox_gt_recentered

    def scale(self, image):
        """TODO: Docstring for scale.
        :returns: TODO

        """
        height = image.shape[0]
        width = image.shape[1]

        self.x1 = self.x1 / width
        self.y1 = self.y1 / height
        self.x2 = self.x2 / width
        self.y2 = self.y2 / height

        self.x1 = self.x1 * self.kScaleFactor
        self.y1 = self.y1 * self.kScaleFactor
        self.x2 = self.x2 * self.kScaleFactor
        self.y2 = self.y2 * self.kScaleFactor

    def get_width(self):
        """TODO: Docstring for get_width.
        :returns: TODO

        """
        return (self.x2 - self.x1)

    def get_height(self):
        """TODO: Docstring for get_width.
        :returns: TODO

        """
        return (self.y2 - self.y1)

    def shift(self, image, lambda_scale_frac, lambda_shift_frac, min_scale, max_scale, shift_motion_model, bbox_rand):
        """TODO: Docstring for shift.
        :returns: TODO

        """
        width = self.get_width()
        height = self.get_height()

        center_x = self.get_center_x()
        center_y = self.get_center_y()

        kMaxNumTries = 10

        new_width = -1
        num_tries_width = 0
        while ((new_width < 0) or (new_width > image.shape[1] - 1)) and (num_tries_width < kMaxNumTries):
            if shift_motion_model:
                width_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sides(lambda_scale_frac)))
            else:
                rand_num = sample_rand_uniform()
                width_scale_factor = rand_num * (max_scale - min_scale) + min_scale

            new_width = width * (1 + width_scale_factor)
            new_width = max(1.0, min((image.shape[1] - 1), new_width))
            num_tries_width = num_tries_width + 1


        new_height = -1
        num_tries_height = 0
        while ((new_height < 0) or (new_height > image.shape[0] - 1)) and (num_tries_height < kMaxNumTries):
            if shift_motion_model:
                height_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sides(lambda_scale_frac)))
            else:
                rand_num = sample_rand_uniform()
                height_scale_factor = rand_num * (max_scale - min_scale) + min_scale

            new_height = height * ( 1 + height_scale_factor )
            new_height = max(1.0, min((image.shape[0] - 1), new_height))
            num_tries_height = num_tries_height + 1


        first_time_x = True
        new_center_x = -1
        num_tries_x = 0

        while (first_time_x or (new_center_x < center_x - width * self.kContextFactor / 2)
                or (new_center_x > center_x + width * self.kContextFactor / 2)
                or ((new_center_x - new_width / 2) < 0)
                or ((new_center_x + new_width / 2) > image.shape[1])
                and (num_tries_x < kMaxNumTries)):

            if shift_motion_model:
                new_x_temp = center_x + width * sample_exp_two_sides(lambda_shift_frac)
            else:
                rand_num = sample_rand_uniform()
                new_x_temp = center_x + rand_num * (2 * new_width) - new_width

            new_center_x = min(image.shape[1] - new_width / 2, max(new_width / 2, new_x_temp))
            first_time_x = False
            num_tries_x = num_tries_x + 1

        first_time_y = True
        new_center_y = -1
        num_tries_y = 0

        while (first_time_y or (new_center_y < center_y - height * self.kContextFactor / 2)
                or (new_center_y > center_y + height * self.kContextFactor / 2)
                or ((new_center_y - new_height / 2) < 0)
                or ((new_center_y + new_height / 2) > image.shape[0])
                and (num_tries_y < kMaxNumTries)):

            if shift_motion_model:
                new_y_temp = center_y + height * sample_exp_two_sides(lambda_shift_frac)
            else:
                rand_num = sample_rand_uniform()
                new_y_temp = center_y + rand_num * (2 * new_height) - new_height

            new_center_y = min(image.shape[0] - new_height / 2, max(new_height / 2, new_y_temp))
            first_time_y = False
            num_tries_y = num_tries_y + 1

        bbox_rand.x1 = new_center_x - new_width / 2
        bbox_rand.x2 = new_center_x + new_width / 2
        bbox_rand.y1 = new_center_y - new_height / 2
        bbox_rand.y2 = new_center_y + new_height / 2

        return bbox_rand


class annotation:

    """Docstring for annotation. """

    def __init__(self):
        """TODO: to be defined1. """
        self.bbox = BoundingBox(0, 0, 0, 0)
        self.image_path = []
        self.disp_width = 0
        self.disp_height = 0

    def setbbox(self, x1, x2, y1, y2):
        """TODO: Docstring for setbbox.
        :returns: TODO

        """
        self.bbox.x1 = x1
        self.bbox.x2 = x2
        self.bbox.y1 = y1
        self.bbox.y2 = y2

    def setImagePath(self, img_path):
        """TODO: Docstring for setImagePath.
        :returns: TODO

        """
        self.image_path = img_path

    def setWidthHeight(self, disp_width, disp_height):
        """TODO: Docstring for setWidthHeight.
        :returns: TODO

        """
        self.disp_width = disp_width
        self.disp_height = disp_height


def load_annotation_file(annotation_file):
        """TODO: Docstring for load_annotation_file.
        :returns: TODO

        """
        list_of_annotations = []
        num_annotations = 0
        root = ET.parse(annotation_file).getroot()
        folder = root.find('folder').text
        if folder[0]!='n':
            folder = 'n' + folder
        print 'loading folder',folder,'....'
        filename = root.find('filename').text
        size = root.find('size')
        disp_width = int(size.find('width').text)
        disp_height = int(size.find('height').text)

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)

            width = xmax - xmin
            height = ymax - ymin

            if width > (kMaxRatio * disp_width) or height > (kMaxRatio * disp_height):
                continue

            if ((xmin < 0) or (ymin < 0) or (xmax <= xmin) or (ymax <= ymin)):
                continue

            # print xmin,xmax,ymin,ymax

            objAnnotation = annotation()
            objAnnotation.setbbox(xmin, xmax, ymin, ymax)
            objAnnotation.setWidthHeight(disp_width, disp_height)
            objAnnotation.setImagePath(os.path.join(folder, filename))
            list_of_annotations.append(objAnnotation)
            num_annotations = num_annotations + 1

        return list_of_annotations, num_annotations


def sample_exp_two_sides(lambda_):
    """TODO: Docstring for sample_exp_two_sides.
    :returns: TODO

    """

    pos_or_neg = random.randint(0, RAND_MAX)
    if (pos_or_neg % 2) == 0:
        pos_or_neg = 1
    else:
        pos_or_neg = -1

    rand_uniform = (random.randint(0, RAND_MAX) + 1) * 1.0 / (RAND_MAX + 2)

    return math.log(rand_uniform) / (lambda_ * pos_or_neg)


def cropPadImage(bbox_tight, image):
    """TODO: Docstring for cropPadImage.
    :returns: TODO

    """
    pad_image_location = computeCropPadImageLocation(bbox_tight, image)
    roi_left = min(pad_image_location.x1, (image.shape[1] - 1))
    roi_bottom = min(pad_image_location.y1, (image.shape[0] - 1))
    roi_width = min(image.shape[1], max(1.0, math.ceil(pad_image_location.x2 - pad_image_location.x1)))
    roi_height = min(image.shape[0], max(1.0, math.ceil(pad_image_location.y2 - pad_image_location.y1)))

    err = 0.000000001 # To take care of floating point arithmetic errors
    cropped_image = image[int(roi_bottom + err):int(roi_bottom + roi_height), int(roi_left + err):int(roi_left + roi_width)]

    # Padded output width and height
    output_width = max(math.ceil(bbox_tight.compute_output_width()), roi_width)
    output_height = max(math.ceil(bbox_tight.compute_output_height()), roi_height)

    if image.ndim > 2:
        output_image = np.zeros((int(output_height), int(output_width), image.shape[2]), dtype=image.dtype)
    else:
        output_image = np.zeros((int(output_height), int(output_width)), dtype=image.dtype)

    # Center of the bounding box
    edge_spacing_x = min(bbox_tight.edge_spacing_x(), (image.shape[1] - 1))
    edge_spacing_y = min(bbox_tight.edge_spacing_y(), (image.shape[0] - 1))

    # rounding should be done to match the width and height
    output_image[int(edge_spacing_y):int(edge_spacing_y) + cropped_image.shape[0], int(edge_spacing_x):int(edge_spacing_x) + cropped_image.shape[1]] = cropped_image
    return output_image, pad_image_location, edge_spacing_x, edge_spacing_y


def computeCropPadImageLocation(bbox_tight, image):
    """TODO: Docstring for computeCropPadImageLocation.
    :returns: TODO

    """
    # Center of the bounding box
    bbox_center_x = bbox_tight.get_center_x()
    bbox_center_y = bbox_tight.get_center_y()

    image_height = image.shape[0]
    image_width = image.shape[1]

    # Padded output width and height
    output_width = bbox_tight.compute_output_width()
    output_height = bbox_tight.compute_output_height()

    roi_left = max(0.0, bbox_center_x - (output_width / 2.))
    roi_bottom = max(0.0, bbox_center_y - (output_height / 2.))

    # Padded roi width
    left_half = min(output_width / 2., bbox_center_x)
    right_half = min(output_width / 2., image_width - bbox_center_x)
    roi_width = max(1.0, left_half + right_half)

    # Padded roi height
    top_half = min(output_height / 2., bbox_center_y)
    bottom_half = min(output_height / 2., image_height - bbox_center_y)
    roi_height = max(1.0, top_half + bottom_half)

    # Padded image location in the original image
    objPadImageLocation = BoundingBox(roi_left, roi_bottom, roi_left + roi_width, roi_bottom + roi_height)

    return objPadImageLocation

def preprocess(image):
        """TODO: Docstring for preprocess.

        :arg1: TODO
        :returns: TODO

        """
        num_channels = channel_resize
        if num_channels == 1 and image.shape[2] == 3:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif num_channels == 1 and image.shape[2] == 4:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif num_channels == 3 and image.shape[2] == 4:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif num_channels == 3 and image.shape[2] == 2:
            image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_out = image

        if image_out.shape != (height_resize, width_resize, channel_resize):
            image_out = cv2.resize(image_out, (width_resize, height_resize), interpolation=cv2.INTER_CUBIC)

        image_out = np.float32(image_out)
        image_out -= np.array([104, 117, 123])
        # image_out = np.transpose(image_out, [2, 0, 1])
        return image_out


def preprocess_caffe(image):
        """TODO: Docstring for preprocess.

        :arg1: TODO
        :returns: TODO

        """
        num_channels = channel_resize
        if num_channels == 1 and image.shape[2] == 3:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif num_channels == 1 and image.shape[2] == 4:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif num_channels == 3 and image.shape[2] == 4:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif num_channels == 3 and image.shape[2] == 2:
            image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_out = image

        if image_out.shape != (height_resize, width_resize, channel_resize):
            image_out = cv2.resize(image_out, (width_resize, height_resize), interpolation=cv2.INTER_CUBIC)

        image_out = np.float32(image_out)
        image_out -= np.array([104, 117, 123])
        image_out = np.transpose(image_out, [2, 0, 1])
        return image_out
