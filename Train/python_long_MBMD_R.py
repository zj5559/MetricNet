import cv2
import os
import sys
sys.path.append('./lib')
sys.path.append('./lib/slim')
# from region_to_bbox import region_to_bbox
# import time
import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from core.model_builder import build_man_model
from object_detection.core import box_list
from object_detection.core import box_list_ops
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import scipy.io as sio
import random
from me_sample_generator import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou


def get_configs_from_pipeline_file(config_file):
  """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.
  Reads training config from file specified by pipeline_config_path flag.
  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(config_file, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  model_config = pipeline_config.model.ssd
  train_config = pipeline_config.train_config
  input_config = pipeline_config.train_input_reader
  eval_config = pipeline_config.eval_config

  return model_config, train_config, input_config, eval_config


def restore_model(sess, model_scope, checkpoint_path, variables_to_restore):
    # variables_to_restore = tf.global_variables()
    name_to_var_dict = dict([(var.op.name.lstrip(model_scope+'/'), var) for var in variables_to_restore
                             if not var.op.name.endswith('Momentum')])
    saver = tf.train.Saver(name_to_var_dict)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, latest_checkpoint)

def crop_search_region(img, gt, win_size, scale=4, mean_rgb=128, offset=None):
    # gt: [ymin, xmin, ymax, xmax]
    bnd_ymin, bnd_xmin, bnd_ymax, bnd_xmax = gt
    bnd_w = bnd_xmax - bnd_xmin
    bnd_h = bnd_ymax - bnd_ymin
    # cx, cy = gt[:2] + gt[2:] / 2
    cy, cx = (bnd_ymin + bnd_ymax)/2, (bnd_xmin+bnd_xmax)/2
    diag = np.sum( bnd_h** 2 + bnd_w**2) ** 0.5
    origin_win_size = diag * scale
    origin_win_size_h, origin_win_size_w = bnd_h * scale, bnd_w * scale
    # origin_win_size_h = origin_win_size
    # origin_win_size_w = origin_win_size
    im_size = img.size[1::-1]
    min_x = np.round(cx - origin_win_size_w / 2).astype(np.int32)
    max_x = np.round(cx + origin_win_size_w / 2).astype(np.int32)
    min_y = np.round(cy - origin_win_size_h / 2).astype(np.int32)
    max_y = np.round(cy + origin_win_size_h / 2).astype(np.int32)
    if offset is not None:
        min_offset_y, max_offset_y = (bnd_ymax - max_y, bnd_ymin - min_y)
        min_offset_x, max_offset_x = (bnd_xmax - max_x, bnd_xmin - min_x)
        offset[0] = np.clip(offset[0] * origin_win_size_h, min_offset_y, max_offset_y)
        offset[1] = np.clip(offset[1] * origin_win_size_w, min_offset_x, max_offset_x)
        offset = np.int32(offset)
        min_y += offset[0]
        max_y += offset[0]
        min_x += offset[1]
        max_x += offset[1]

    win_loc = np.array([min_y, min_x])
    gt_x_min, gt_y_min = ((bnd_xmin-min_x)/origin_win_size_w, (bnd_ymin - min_y)/origin_win_size_h) #coordinates on window
    gt_x_max, gt_y_max = [(bnd_xmax-min_x)/origin_win_size_w, (bnd_ymax - min_y)/origin_win_size_h] #relative coordinates of gt bbox to the search region

    unscaled_w, unscaled_h = [max_x - min_x + 1, max_y - min_y + 1]
    min_x_win, min_y_win, max_x_win, max_y_win = (0, 0, unscaled_w, unscaled_h)
    min_x_im, min_y_im, max_x_im, max_y_im = (min_x, min_y, max_x+1, max_y+1)

    img = img.crop([min_x_im, min_y_im, max_x_im, max_y_im])
    img_array = np.array(img)

    if min_x < 0:
        min_x_im = 0
        min_x_win = 0 - min_x
    if min_y < 0:
        min_y_im = 0
        min_y_win = 0 - min_y
    if max_x+1 > im_size[1]:
        max_x_im = im_size[1]
        max_x_win = unscaled_w - (max_x + 1 - im_size[1])
    if max_y+1 > im_size[0]:
        max_y_im = im_size[0]
        max_y_win = unscaled_h- (max_y +1 - im_size[0])

    unscaled_win = np.ones([unscaled_h, unscaled_w, 3], dtype=np.uint8) * np.uint8(mean_rgb)
    unscaled_win[min_y_win:max_y_win, min_x_win:max_x_win] = img_array[min_y_win:max_y_win, min_x_win:max_x_win]

    unscaled_win = Image.fromarray(unscaled_win)
    height_scale, width_scale = np.float32(unscaled_h)/win_size, np.float32(unscaled_w)/win_size
    win = unscaled_win.resize([win_size, win_size], resample=Image.BILINEAR)
    # win = sp.misc.imresize(unscaled_win, [win_size, win_size])
    return win, np.array([gt_y_min, gt_x_min, gt_y_max, gt_x_max]), win_loc, [height_scale, width_scale]
    # return win, np.array([gt_x_min, gt_y_min, gt_x_max, gt_y_max]), diag, np.array(win_loc)
def build_init_graph(model, model_scope, reuse=None):
    input_init_image = tf.placeholder(dtype=tf.uint8, shape=[128,128,3])
    float_init_image = tf.to_float(input_init_image)
    float_init_image = tf.expand_dims(tf.expand_dims(float_init_image, axis=0), axis=0)
    preprocessed_init_image = model.preprocess(float_init_image, [128,128])
    with tf.variable_scope(model_scope, reuse=reuse):
        init_feature_maps = model.extract_init_feature(preprocessed_init_image)
    return init_feature_maps,input_init_image

def build_box_predictor(model, model_scope,init_feature_maps,reuse=None):
    input_cur_image = tf.placeholder(dtype=tf.uint8, shape=[300, 300, 3])
    images = tf.expand_dims(input_cur_image, axis=0)
    float_images = tf.to_float(images)
    preprocessed_images = model.preprocess(float_images)
    preprocessed_images = tf.expand_dims(preprocessed_images, axis=0)
    input_init_gt_box = tf.constant(np.zeros((1, 4)), dtype=tf.float32)
    init_gt_box = tf.reshape(input_init_gt_box, shape=[1,1,4])
    groundtruth_classes = tf.ones(dtype=tf.float32, shape=[1, 1, 1])
    model.provide_groundtruth(init_gt_box,
                              groundtruth_classes,
                              None)
    with tf.variable_scope(model_scope, reuse=reuse):
        prediction_dict = model.predict_box_with_init(init_feature_maps, preprocessed_images, istraining=False)

    detections = model.postprocess(prediction_dict)
    original_image_shape = tf.shape(preprocessed_images)
    absolute_detection_boxlist = box_list_ops.to_absolute_coordinates(
        box_list.BoxList(tf.squeeze(detections['detection_boxes'], axis=0)),
        original_image_shape[2], original_image_shape[3])
    return absolute_detection_boxlist.get(), detections['detection_scores'], input_cur_image

class MobileTracker(object):
    def __init__(self):
        init_training = True
        config_file = './model/ssd_mobilenet_tracking.config'
        checkpoint_dir = './model/dump'

        model_config, train_config, input_config, eval_config = get_configs_from_pipeline_file(config_file)
        model = build_man_model(model_config=model_config, is_training=False)
        model_scope = 'model'
        self.initFeatOp, self.initInputOp = build_init_graph(model, model_scope, reuse=None)
        self.initConstantOp = tf.placeholder(tf.float32, [1,1,1,512])
        self.pre_box_tensor, self.scores_tensor, self.input_cur_image = build_box_predictor(model, model_scope, self.initConstantOp, reuse=None)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #if not init_training:
        variables_to_restore = tf.global_variables()
        restore_model(self.sess, model_scope, checkpoint_dir, variables_to_restore)

    def initialize(self, image, bbox):
        init_img = Image.fromarray(image)
        init_gt = [bbox[1], bbox[0], bbox[1]+bbox[3], bbox[0]+bbox[2]] # ymin xmin ymax xmax
        init_img_array = np.array(init_img)
        self.expand_channel = False
        if init_img_array.ndim < 3:
            init_img_array = np.expand_dims(init_img_array, axis=2)
            init_img_array = np.repeat(init_img_array, repeats=3, axis=2)
            init_img = Image.fromarray(init_img_array)
            self.expand_channel = True

        gt_boxes = np.zeros((1,4))
        gt_boxes[0,0] = init_gt[0] / float(init_img.height)
        gt_boxes[0,1] = init_gt[1] / float(init_img.width)
        gt_boxes[0,2] = init_gt[2] / float(init_img.height)
        gt_boxes[0,3] = init_gt[3] / float(init_img.width)

        img1_xiaobai = np.array(init_img)
        pad_x = 36.0 / 264.0 * (gt_boxes[0, 3] - gt_boxes[0, 1]) * init_img.width
        pad_y = 36.0 / 264.0 * (gt_boxes[0, 2] - gt_boxes[0, 0]) * init_img.height
        startx = gt_boxes[0, 1] * init_img.width - pad_x
        starty = gt_boxes[0, 0] * init_img.height - pad_y
        endx = gt_boxes[0, 3] * init_img.width + pad_x
        endy = gt_boxes[0, 2] * init_img.height + pad_y
        left_pad = max(0, int(-startx))
        top_pad = max(0, int(-starty))
        right_pad = max(0, int(endx - init_img.width + 1))
        bottom_pad = max(0, int(endy - init_img.height + 1))

        startx = int(startx + left_pad)
        starty = int(starty + top_pad)
        endx = int(endx + left_pad)
        endy = int(endy + top_pad)

        if top_pad or left_pad or bottom_pad or right_pad:
            r = np.pad(img1_xiaobai[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                       constant_values=128)
            g = np.pad(img1_xiaobai[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                       constant_values=128)
            b = np.pad(img1_xiaobai[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                       constant_values=128)
            r = np.expand_dims(r, 2)
            g = np.expand_dims(g, 2)
            b = np.expand_dims(b, 2)


            img1_xiaobai = np.concatenate((r, g, b), axis=2)
        img1_xiaobai = Image.fromarray(img1_xiaobai)
        im = np.array(init_img)
        # gt_boxes resize
        init_img_crop = img1_xiaobai.crop(np.int32([startx, starty, endx, endy]))
        init_img_crop = init_img_crop.resize([128,128], resample=Image.BILINEAR)
        self.last_gt = init_gt

        self.init_img_array = np.array(init_img_crop)
        self.init_feature_maps = self.sess.run(self.initFeatOp, feed_dict={self.initInputOp:self.init_img_array})

    def get_search_list(self, cur_ori_img, gt):
        target_h = gt[2]
        target_w = gt[3]
        search_list = []
        self.startx = 2 * target_w
        self.starty = 2 * target_h
        # whole image
        while (self.startx < cur_ori_img.width - 1) and (self.starty < cur_ori_img.height - 1):
            search_gt = np.int32(
                [self.starty - target_h / 2.0, self.startx - target_w / 2.0,
                 self.starty + target_h / 2.0,
                 self.startx + target_w / 2.0])
            search_list.append(search_gt)

            self.starty = 3.5 * target_h + self.starty
            if self.starty >= cur_ori_img.height - 1 and self.startx < cur_ori_img.width - 1:
                self.starty = 0
                self.startx = 3.5 * target_w + self.startx
        return search_list

    def get_search_list1(self, cur_ori_img, gt):
        target_w = gt[2]
        target_h = gt[3]
        search_list = []
        self.startx = 2 * target_w
        self.starty = 2 * target_h
        # whole image
        while (self.startx < cur_ori_img.width - 1) and (self.starty < cur_ori_img.height - 1):
            search_gt = np.int32(
                [self.starty - target_h / 2.0, self.startx - target_w / 2.0,
                 self.starty + target_h / 2.0,
                 self.startx + target_w / 2.0])
            search_list.append(search_gt)

            self.starty = 3.5 * target_h + self.starty
            if self.starty >= cur_ori_img.height - 1 and self.startx < cur_ori_img.width - 1:
                self.starty = 0
                self.startx = 3.5 * target_w + self.startx
        if len(search_list)==0:
            self.startx = 1.5 * target_w
            self.starty = 1.5 * target_h
            while (self.startx < cur_ori_img.width - 1) and (self.starty < cur_ori_img.height - 1):
                search_gt = np.int32(
                    [self.starty - target_h / 2.0, self.startx - target_w / 2.0,
                     self.starty + target_h / 2.0,
                     self.startx + target_w / 2.0])
                search_list.append(search_gt)

                self.starty = 2.0 * target_h + self.starty
                if self.starty >= cur_ori_img.height - 1 and self.startx < cur_ori_img.width - 1:
                    self.starty = 0
                    self.startx = 2.0 * target_w + self.startx
        return search_list

    def track(self, image, gt):
        cur_ori_img = Image.fromarray(image)

        search_list = self.get_search_list1(cur_ori_img, gt)#change here
        searchLen = len(search_list)

        neg_regions = []
        num_all=0
        for i in range(searchLen):
            search_gt = search_list[i]
            cropped_img1, last_gt_norm1, win_loc1, scale1 = crop_search_region(cur_ori_img, search_gt, 300,
                                                                               mean_rgb=128)
            cur_img_array1 = np.array(cropped_img1)
            detection_box1, scores1 = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                                    feed_dict={self.input_cur_image: cur_img_array1,
                                                               self.initConstantOp: self.init_feature_maps})

            detection_box1[:, 0] = detection_box1[:, 0] * scale1[0] + win_loc1[0]
            detection_box1[:, 1] = detection_box1[:, 1] * scale1[1] + win_loc1[1]
            detection_box1[:, 2] = detection_box1[:, 2] * scale1[0] + win_loc1[0]
            detection_box1[:, 3] = detection_box1[:, 3] * scale1[1] + win_loc1[1]
            detection_box1[:,0] = np.clip(detection_box1[:,0],0, image.shape[0] - 1)
            detection_box1[:,1] = np.clip(detection_box1[:,1],0, image.shape[1] - 1)
            detection_box1[:,2] = np.clip(detection_box1[:,2],0, image.shape[0] - 1)
            detection_box1[:,3] = np.clip(detection_box1[:,3],0, image.shape[1] - 1)
            detection_box = detection_box1.copy()
            detection_box[:,0] = detection_box1[:,1]
            detection_box[:,1] = detection_box1[:,0]
            detection_box[:,2] = detection_box1[:,3] - detection_box1[:,1]
            detection_box[:,3] = detection_box1[:,2] - detection_box1[:,0]

            boxes = []
            for t in range(20):
                iou1 = _compute_iou(detection_box[t,:], gt)
                if iou1 < 0.3 and scores1[0,t] > 0.2 and detection_box[t,2] > 3 and detection_box[t,3] > 3:
                    boxes.append(detection_box[t:(t+1),:])
                if scores1[0,t] < 0.2:
                    break

            if len(boxes) > 0:
                boxes = np.concatenate(boxes, axis=0)
                neg_region = me_extract_regions(image, boxes)
                neg_regions.append(neg_region)
                num_all+=neg_region.shape[0]
            if num_all>20:
                neg_regions = np.concatenate(neg_regions, axis=0)
                return neg_regions

        if len(neg_regions) > 0:
            neg_regions = np.concatenate(neg_regions, axis=0)
        return neg_regions

