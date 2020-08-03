import tensorflow as tf
import numpy as np
from lxml import etree
import os
import PIL.Image as Image
from collections import OrderedDict
from utils.create_imagenet_seq_tf_record import recursive_parse_xml, _bytes_list_feature, _float_list_feature, _int64_list_feature

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import cv2

# image_root = '/media/2TB/Research/DataSet/Tracking/ALOV/imagedata++'
# ann_root = '/media/2TB/Research/DataSet/Tracking/ALOV/alov300++_rectangleAnnotation_full'
# res_path = '/media/2TB/Research/DataSet/Tracking/ALOV/train.record'

image_root = '/home/xiaobai/Documents/ALOV300/imagedata++'
ann_root = '/home/xiaobai/Documents/ALOV300/alov300++_rectangleAnnotation_full'
vid_root = '/home/xiaobai/Documents/'
vid_ann_root = '/home/xiaobai/Documents/'
res_path = '/home/xiaobai/Documents/train_vid.record'


def main(_):
    image_root = '/home/xiaobai/Documents/ILSVRC2015/Data/VID/train/'
    ann_root = '/home/xiaobai/Documents/ILSVRC2015/Annotations/VID/train/'
    folders = os.listdir(image_root)
    folders.sort()
    snippet_list = list()
    for folder in folders:
        snippets = os.listdir(os.path.join(image_root, folder))
        snippets.sort()
        snippet_list.extend([os.path.join(folder,snippet) for snippet in snippets])
    np.random.shuffle(snippet_list)

    writer = tf.python_io.TFRecordWriter(res_path)
    num_snippets = len(snippet_list)
    class_id = 0
    for sid, snippet in enumerate(snippet_list):
        if sid%100 == 0:
            print("On snippet: %d / %d"%(sid, num_snippets))
        frames = os.listdir(os.path.join(image_root, snippet))
        frames.sort()
        object_dict = OrderedDict()
        for frame in frames:
            img = Image.open(os.path.join(image_root, snippet, frame))
            img_array = np.array(img)
            height, width = img_array.shape[0:2]
            if img.format != 'JPEG' or img_array.ndim == 3 and img_array.shape[2] > 3:
                continue

            xml_file = os.path.join(ann_root, snippet, frame.rstrip('JPEG') + 'xml')
            with open(xml_file) as fid:
                xml = etree.fromstring(fid.read())
            xml = recursive_parse_xml(xml)
            if 'object' not in xml:
                continue
            for obj in xml['object']:
                xmin, xmax, ymin, ymax = (
                    float(obj['bndbox']['xmin']) / width, float(obj['bndbox']['xmax']) / width,
                    float(obj['bndbox']['ymin']) / height, float(obj['bndbox']['ymax']) / height)

                if xmin >= xmax or ymin >= ymax or xmin < 0 or xmax > 1 or ymin < 0 or ymax > 1:
                    continue

                if obj['trackid'] not in object_dict.keys():
                    object_dict[obj['trackid']] = dict({'frame': list(),
                                                        'xmax': list(),
                                                        'xmin': list(),
                                                        'ymax': list(),
                                                        'ymin': list()})
                object_dict[obj['trackid']]['frame'].append(snippet + '/' + frame)
                object_dict[obj['trackid']]['xmax'].append(xmax)
                object_dict[obj['trackid']]['xmin'].append(xmin)
                object_dict[obj['trackid']]['ymax'].append(ymax)
                object_dict[obj['trackid']]['ymin'].append(ymin)

        for obj in object_dict.values():
            example = tf.train.Example(features=tf.train.Features(feature={
                'bndbox/xmin': _float_list_feature(obj['xmin']),
                'bndbox/xmax': _float_list_feature(obj['xmax']),
                'bndbox/ymin': _float_list_feature(obj['ymin']),
                'bndbox/ymax': _float_list_feature(obj['ymax']),
                'image_name': _bytes_list_feature(obj['frame']),
                'folder': _bytes_list_feature(['ILSVRC2015']),
                'class_id':_float_list_feature([class_id])}))
            class_id += 1
            writer.write(example.SerializeToString())

    # video_root = '/home/xiaobai/Documents/NUSPRO'
    # snippet_list = list()
    # folders = os.listdir(video_root)
    # folders.sort()
    # np.random.shuffle(folders)
    # num_folders = len(folders)
    # for folder_id, folder in enumerate(folders):
    #     print("On snippet: %d / %d" % (folder_id, num_folders))
    #     folder_path = video_root + '/' + folder
    #     frame_names = os.listdir(folder_path)
    #     frame_names = [frame_name for frame_name in frame_names if frame_name.endswith('jpg')]
    #     frame_names.sort()
    #     occ_info = folder_path + '/' + 'occlusion.txt'
    #     occ_info = np.loadtxt(occ_info)
    #     gt_boxes = folder_path + '/' + 'groundtruth.txt'
    #     gt_boxes = np.loadtxt(gt_boxes)
    #
    #     valid_frames = list()
    #     valid_xmins = list()
    #     valid_ymins = list()
    #     valid_xmaxs = list()
    #     valid_ymaxs = list()
    #     for id in range(len(frame_names)):
    #         if id < len(occ_info):
    #             occ = occ_info[id]
    #         if occ == 1:
    #             continue
    #
    #         frame_name = frame_names[id]
    #         im = cv2.imread(folder_path + '/' + frame_name)
    #         # cv2.rectangle(im,(int(gt_boxes[id,0]),int(gt_boxes[id,1])), (int(gt_boxes[id,2]), int(gt_boxes[id,3])), (0,0,255),2)
    #         # cv2.imshow("haha",im)
    #         # cv2.waitKey(0)
    #         # print"hehe"
    #
    #         xmin = int(gt_boxes[id, 0])
    #         xmax = int(gt_boxes[id, 2])
    #         ymin = int(gt_boxes[id, 1])
    #         ymax = int(gt_boxes[id, 3])
    #
    #         xmin = xmin / float(im.shape[1])
    #         xmax = xmax / float(im.shape[1])
    #         ymin = ymin / float(im.shape[0])
    #         ymax = ymax / float(im.shape[0])
    #
    #         valid_xmins.append(xmin)
    #         valid_xmaxs.append(xmax)
    #         valid_ymins.append(ymin)
    #         valid_ymaxs.append(ymax)
    #         valid_frames.append(folder + '/' + frame_name)
    #
    #         example = tf.train.Example(features=tf.train.Features(feature={
    #             'bndbox/xmin': _float_list_feature(valid_xmins),
    #             'bndbox/xmax': _float_list_feature(valid_xmaxs),
    #             'bndbox/ymin': _float_list_feature(valid_ymins),
    #             'bndbox/ymax': _float_list_feature(valid_ymaxs),
    #             'image_name': _bytes_list_feature(valid_frames),
    #             'folder': _bytes_list_feature(['NUSPRO'])}))
    #         writer.write(example.SerializeToString())
    writer.close()
    print class_id
    print('Create TFRecord Success!')


if __name__ == '__main__':
    tf.app.run()