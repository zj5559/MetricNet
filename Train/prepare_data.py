import cv2
import numpy as np
import os
from me_sample_generator import *
from PIL import Image
from python_long_MBMD_R import MobileTracker

src_path = '/media/zj/4T/Dataset/LaSOT/dataset/images/'
dst_path = '/media/zj/4T/Dataset/LaSOT_crops/'
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

Pos_Per_Frame = 5

tracker = MobileTracker()

data_dict = dict()
folder_list = os.listdir(src_path)
folder_list.sort()
for folder_id, folder in enumerate(folder_list):
    # if folder_id < 42:
    #     continue
    folder_path = src_path + folder
    dst_folder_path = dst_path + folder
    if not os.path.exists(dst_folder_path):
        os.mkdir(dst_folder_path)

    seq_list = os.listdir(folder_path)
    seq_list.sort()
    for seq_id, seq in enumerate(seq_list):
        # if folder_id == 42 and seq_id < 15:
        #     continue
        seq_path = folder_path + '/' + seq
        dst_seq_path = dst_folder_path + '/' + seq
        if not os.path.exists(dst_seq_path):
            os.mkdir(dst_seq_path)
        else:
            continue

        dst_pos_path = dst_folder_path + '/' + seq + '/pos/'
        if not os.path.exists(dst_pos_path):
            os.mkdir(dst_pos_path)

        dst_neg_path = dst_folder_path + '/' + seq + '/neg/'
        if not os.path.exists(dst_neg_path):
            os.mkdir(dst_neg_path)

        img_list = sorted([folder + '/' + seq + '/img/' + p for p in os.listdir(seq_path + '/img/') if
                           os.path.splitext(p)[1] == '.jpg'])
        gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')
        gt[:,0] = gt[:,0] - 1.0
        gt[:,1] = gt[:,1] - 1.0
        gt = gt.astype(np.int32)
        occ = np.loadtxt(seq_path + '/full_occlusion.txt', delimiter=',')
        out_of_view = np.loadtxt(seq_path + '/out_of_view.txt', delimiter=',')
        start_id = 0

        img = Image.open(src_path + img_list[0])
        tracker.initialize(np.array(img), gt[0])
        img_num = len(img_list)
        neg_start_id = 0
        for im_id, im_name in enumerate(img_list):
            if np.mod(im_id, 3) != 0:
                continue
            #print(folder_id, seq_id, im_id,'/',img_num)
            print(folder_id, seq_id, im_name, img_num)
            if gt[im_id,2] < 3 or gt[im_id,3] < 3:
                continue
            neg_regions = tracker.track(np.array(img), gt[im_id])
            if len(neg_regions) > 0:
                for i in range(len(neg_regions)):
                    save_name = dst_neg_path + '%08d.jpg'%neg_start_id
                    tmp = (neg_regions[i][:,:,::-1]).copy()
                    cv2.imwrite(save_name, tmp)
                    neg_start_id += 1
                    # cv2.namedWindow("neg", cv2.WINDOW_NORMAL)
                    # tmp = (neg_regions[i][:,:,::-1]).copy()
                    # cv2.imshow("neg", tmp)
                    # cv2.waitKey(0)

            if occ[im_id] == 1 or out_of_view[im_id] == 1:
                continue
            img = Image.open(src_path + im_name)
            pos_examples = gen_samples(SampleGenerator('gaussian', img.size, 0.1, 1.2), gt[im_id], Pos_Per_Frame, [0.7, 1])
            pos_region = me_extract_regions(np.array(img), pos_examples)
            pos_region = (pos_region[:,:,:,::-1]).copy()
            if pos_region.shape[0] == Pos_Per_Frame:
                for i in range(Pos_Per_Frame):
                    save_name = dst_pos_path + '%08d.jpg'%start_id
                    cv2.imwrite(save_name, pos_region[i])
                    start_id += 1


        # seq_name = folder + '/' + seq
        # data_dict[seq_name] = dict()
        # data_dict[seq_name]['img_list'] = img_list
        # gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')
        # occ = np.loadtxt(seq_path + '/full_occlusion.txt', delimiter=',')
        # out_of_view = np.loadtxt(seq_path + '/out_of_view.txt', delimiter=',')
        # data_dict[seq_name]['occ'] = occ
        # data_dict[seq_name]['out_of_view'] = out_of_view
        # data_dict[seq_name]['img_num'] = len(img_list)
        # data_dict[seq_name]['gt'] = gt

