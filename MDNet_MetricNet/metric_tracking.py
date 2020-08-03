#coding=utf-8
import cv2 as cv
import os
# from region_to_bbox import region_to_bbox
import time
import tensorflow as tf
import yaml, json
import numpy as np
base_path =os.getcwd()
import sys
sys.path.append(os.path.join(base_path, 'implementation'))
sys.path.append(os.path.join(base_path, 'pyMDNet/modules'))
sys.path.append(os.path.join(base_path, 'pyMDNet/tracking'))
# pymdnet
from pyMDNet.modules.model import *
sys.path.insert(0, os.path.join(base_path, 'pyMDNet'))
from pyMDNet.modules.model import MDNet, BCELoss, set_optimizer
from pyMDNet.modules.sample_generator import SampleGenerator
from PIL import Image
from pyMDNet.modules.utils import overlap_ratio
from pyMDNet.tracking.data_prov import RegionExtractor
from pyMDNet.tracking.run_tracker import *
from pyMDNet.modules.judge_metric import *
from pyMDNet.tracking.bbreg import BBRegressor
from pyMDNet.tracking.gen_config import gen_config
from implementation.otbdataset import *
from implementation.uavdataset import *
import xlwt
from tracking_utils import _init_video, _compile_results, show_res
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from numpy.random import seed
import torch
from torch.autograd import Variable
# from tensorflow import set_random_seed
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
# print('before opts1')
opts = yaml.safe_load(open(os.path.join(base_path,'pyMDNet/tracking/options_otb.yaml'),'r'))
# print('after opts1')

class Region:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
def lof_fit(data,k=5,method='l2'):
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', metric=method, contamination=0.1)
    clf.fit(data)
    return clf
def lof(predict, clf,k=5, method='l2',thresh=2):
    # calculating LOF
    predict = -clf._score_samples(predict)
    # predict=predict[200:]
    # identifying outliers
    result=predict<=thresh
    return result

class metric_tracker(object):
    def __init__(self, image, region, imagefile=None, video=None, p=None, groundtruth=None):
        np.random.seed(0)
        torch.manual_seed(0)
        self.i = 0
        self.p = p
        self.video=video
        if groundtruth is not None:
            self.groundtruth = groundtruth
        else:
            self.groundtruth=None
        init_gt1 = [region.x, region.y, region.width, region.height]
        if self.p.base_tracker == 'pymdnet':
            self.init_pymdnet(image, init_gt1)
        elif self.p.base_tracker == 'metricnet':
            self.init_metricnet(image, init_gt1)

    def init_metricnet(self, image, init_bbox):
        target_bbox = np.array(init_bbox)
        self.last_result = target_bbox
        self.pymodel = MDNet(os.path.join(base_path, 'pyMDNet/models/mdnet_imagenet_vid.pth'))
        if opts['use_gpu']:
            self.pymodel = self.pymodel.cuda()
        self.pymodel.set_learnable_params(opts['ft_layers'])

        # Init criterion and optimizer
        self.criterion = BCELoss()
        init_optimizer = set_optimizer(self.pymodel, opts['lr_init'], opts['lr_mult'])
        self.update_optimizer = set_optimizer(self.pymodel, opts['lr_update'], opts['lr_mult'])

        tic = time.time()
        # metric
        self.metric_model = model_load(opts['metric_model'])
        #warmup
        tmp=np.random.rand(5,3,107,107)
        tmp = torch.Tensor(tmp)
        tmp = (Variable(tmp)).type(torch.FloatTensor).cuda()
        self.metric_model.eval()
        tmp =self.metric_model(tmp)


        self.target_metric_feature = get_target_feature(self.metric_model, target_bbox, np.array(image))
        self.target_metric_feature_all = []
        self.target_metric_feature_all.append(self.target_metric_feature)

        # Draw pos/neg samples
        pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
            target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        neg_examples = np.concatenate([
            SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
            SampleGenerator('whole', image.size)(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
        # print(neg_examples)
        neg_examples = np.random.permutation(neg_examples)
        #metric
        ii=0
        self.pos_all=np.zeros(pos_examples.shape[0])
        self.pos_all_feature=np.zeros((pos_examples.shape[0],1024))
        while ii<pos_examples.shape[0]:
            with torch.no_grad():
                pos_metric_feature,pos_metric_dist = get_metric_dist_lof(self.metric_model, pos_examples[ii:ii+50], np.array(image),self.target_metric_feature, opts)
            self.pos_all[ii:ii+50]=pos_metric_dist.cpu().detach().numpy()
            self.pos_all_feature[ii:ii+50]=pos_metric_feature.cpu().detach().numpy()
            ii=ii+50
        self.pos_feature_center =torch.from_numpy(np.mean(self.pos_all_feature,axis=0).reshape((1, 1024))).float().cuda()
        self.clf=lof_fit(self.pos_all_feature[0:opts['n_pos_update']],k=opts['pos_k'],method=opts['method'])
        del pos_metric_feature,pos_metric_dist
        torch.cuda.empty_cache()
        opts['pos_thresh'] = self.pos_all.max() * opts['pos_rate']  # 2.5
        opts['metric_similar_thresh'] = self.pos_all.mean() * opts['similar_rate']
        # print('pos_thresh is:', opts['pos_thresh'])
        # print('similar_thresh is:', opts['metric_similar_thresh'])

        # Extract pos/neg features
        pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.feat_dim = pos_feats.size(-1)

        # Initial training
        train(self.pymodel, self.criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], opts=opts)
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
                                         opts['aspect_bbreg'])(
            target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
        bbreg_feats = forward_samples(self.pymodel, image, bbreg_examples, opts)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
        del bbreg_feats
        torch.cuda.empty_cache()
        # Init sample generators
        self.sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
        self.pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

        # Init pos/neg features for update
        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]

        samples = self.sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)

        self.top_scores,_= sample_scores[:, 1].topk(5)
        self.spf_total = 0


    def init_pymdnet(self, image, init_bbox):
        print('This is mdnet')
        target_bbox = np.array(init_bbox)
        self.last_result = target_bbox
        self.pymodel = MDNet(os.path.join(base_path, 'pyMDNet/models/mdnet_imagenet_vid.pth'))
        if opts['use_gpu']:
            self.pymodel = self.pymodel.cuda()
        self.pymodel.set_learnable_params(opts['ft_layers'])

        # Init criterion and optimizer
        self.criterion = BCELoss()
        init_optimizer = set_optimizer(self.pymodel, opts['lr_init'], opts['lr_mult'])
        self.update_optimizer = set_optimizer(self.pymodel, opts['lr_update'], opts['lr_mult'])

        tic = time.time()

        # Draw pos/neg samples
        pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
            target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        neg_examples = np.concatenate([
            SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
            SampleGenerator('whole', image.size)(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
        neg_examples = np.random.permutation(neg_examples)

        # Extract pos/neg features
        pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.feat_dim = pos_feats.size(-1)

        # Initial training
        train(self.pymodel, self.criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], opts=opts)
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
                                         opts['aspect_bbreg'])(
            target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
        bbreg_feats = forward_samples(self.pymodel, image, bbreg_examples, opts)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
        del bbreg_feats
        torch.cuda.empty_cache()
        # Init sample generators
        self.sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
        self.pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

        # Init pos/neg features for update
        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]

        spf_total = time.time() - tic
    def metricnet_track(self, image):
        tic=time.time()
        self.i += 1
        self.image = image

        target_bbox = self.last_result
        samples = self.sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > 0 #and top_dist[9]<opts['pos_thresh']
        with torch.no_grad():
            self.target_metric_feature_tmp = get_target_feature(self.metric_model, target_bbox, np.array(image))
            success1, target_dist = judge_success(self.metric_model, self.target_metric_feature_tmp, self.target_metric_feature, opts)
        if success:
            success=success1
        # Expand search area at failure
        if success:
            self.sample_generator.set_trans(opts['trans'])
        else:
            self.sample_generator.expand_trans(opts['trans_limit'])

        self.last_result = target_bbox
        # Bbox regression
        bbreg_bbox = self.pymdnet_bbox_reg(success, samples, top_idx)

        # Save result
        region = bbreg_bbox

        # Data collect
        if success:
            self.collect_samples_metricnet(image)

        # Short term update
        if not success:
            self.pymdnet_short_term_update()

        # Long term update
        elif self.i % opts['long_interval'] == 0:
            self.pymdnet_long_term_update()

        self.spf_total = self.spf_total+time.time() - tic
        return region

    def pymdnet_track(self, image):
        self.i += 1
        self.image = image
        target_bbox = self.last_result
        samples = self.sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > 0
        # Expand search area at failure
        if success:
            self.sample_generator.set_trans(opts['trans'])
        else:
            self.sample_generator.expand_trans(opts['trans_limit'])

        self.last_result = target_bbox
        # Bbox regression
        bbreg_bbox = self.pymdnet_bbox_reg(success, samples, top_idx)

        # Save result
        region = bbreg_bbox

        # Data collect
        if success:
            self.collect_samples_pymdnet()

        # Short term update
        if not success:
            self.pymdnet_short_term_update()

        # Long term update
        elif self.i % opts['long_interval'] == 0:
            self.pymdnet_long_term_update()

        return region
    def collect_samples_metricnet(self,image):
        target_bbox = self.last_result
        pos_examples = self.pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
        #metric
        #pos_samples use lof to filter
        with torch.no_grad():
            # pos_examples = judge_metric_lof(self.metric_model, pos_examples, np.array(image), self.target_metric_feature, self.clf_pos,opts)
            pos_features = get_anchor_feature(self.metric_model, np.array(image), pos_examples)  # anchor_box: (1,4) x,y,w,h
            pos_features = pos_features.cpu().detach().numpy()
        # result=lof(self.pos_all_feature[0:opts['n_pos_update']],pos_features,k=opts['pos_k'],method=opts['method'],thresh=opts['pos_thresh_lof'])
        result=lof(pos_features,self.clf,thresh=opts['pos_thresh_lof'])
        pos_examples = pos_examples[result]

        if pos_examples.shape[0]>0:
            pos_feats = forward_samples(self.pymodel, self.image, pos_examples, opts)
            with torch.no_grad():
                dist_tmp = get_metric_dist_by_feature(self.metric_model, self.target_metric_feature_all,self.target_metric_feature_tmp, opts)
            idx_tmp = 0
            for idx in range(dist_tmp.shape[0]):
                if dist_tmp[idx] < opts['metric_similar_thresh']:
                    self.target_metric_feature_all.pop(idx - idx_tmp)
                    self.pos_feats_all.pop(idx - idx_tmp)
                    idx_tmp = idx_tmp + 1
            self.pos_feats_all.append(pos_feats)
            self.target_metric_feature_all.append(self.target_metric_feature_tmp)
        if len(self.pos_feats_all) > opts['n_frames_long']:
            del self.pos_feats_all[0]
            del self.target_metric_feature_all[0]

        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
        with torch.no_grad():
            # print(neg_examples)
            result=judge_metric_center(self.metric_model,neg_examples,np.array(image),self.pos_feature_center,opts)
        neg_examples = neg_examples[result]
        if neg_examples.shape[0] > 0:
            neg_feats = forward_samples(self.pymodel, self.image, neg_examples, opts)
            self.neg_feats_all.append(neg_feats)
        if len(self.neg_feats_all) > opts['n_frames_short']:
            del self.neg_feats_all[0]
    def collect_samples_pymdnet(self):
        target_bbox = self.last_result
        pos_examples = self.pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])

        pos_feats = forward_samples(self.pymodel, self.image, pos_examples, opts)
        self.pos_feats_all.append(pos_feats)
        if len(self.pos_feats_all) > opts['n_frames_long']:
            del self.pos_feats_all[0]

        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
        neg_feats = forward_samples(self.pymodel, self.image, neg_examples, opts)
        self.neg_feats_all.append(neg_feats)
        if len(self.neg_feats_all) > opts['n_frames_short']:
            del self.neg_feats_all[0]

    def pymdnet_short_term_update(self):
        # Short term update
        nframes = min(opts['n_frames_short'], len(self.pos_feats_all))
        pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
        neg_data = torch.cat(self.neg_feats_all, 0)
        train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
              opts=opts)
    def pymdnet_long_term_update(self):
        # Short term update
        pos_data = torch.cat(self.pos_feats_all, 0)
        neg_data = torch.cat(self.neg_feats_all, 0)
        train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
              opts=opts)
    def metric_bbox_reg(self, success,bbreg_samples):
        target_bbox = self.last_result
        if success:
            bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(self.pymodel, self.image, bbreg_samples, opts)
            bbreg_bbox = self.bbreg.predict(bbreg_feats, bbreg_samples)
        else:
            bbreg_bbox = target_bbox
        return bbreg_bbox

    def pymdnet_bbox_reg(self, success, samples, top_idx):
        target_bbox = self.last_result
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(self.pymodel, self.image, bbreg_samples, opts)
            bbreg_samples = self.bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox
        return bbreg_bbox

    def tracking(self, image):
        self.i += 1
def relate_axis(target_box,im_size):
    #target_box=target_box1
    target_box[2]+=target_box[0]
    target_box[3]+=target_box[1]
    target_box[0]=target_box[0]/im_size[0]
    target_box[2]=target_box[2]/im_size[0]
    target_box[1]=target_box[1]/im_size[1]
    target_box[3]=target_box[3]/im_size[1]
    return target_box
def eval_tracking(Dataset, video_spe=None, save=False, p=None):
    if Dataset == 'otb':
        data_dir = '/media/zj/4T/Dataset/OTB-100'
    elif Dataset == "lasot":
        data_dir = '/media/zj/4T/Dataset/LaSOT/dataset/images'
        tmp = video_spe.split('-')
        data_dir = os.path.join(data_dir, tmp[0])
    elif Dataset == 'uav123':
        data_dir = '/media/zj/4T/Dataset/UAV123/Dataset_UAV123/UAV123'

    if video_spe is not None:
        sequence_list = [video_spe]
    else:
        sequence_list = os.listdir(data_dir)
        sequence_list.sort()
        sequence_list = [title for title in sequence_list if not title.endswith("txt")]
    base_save_path = p.save_path
    for seq_id, video in enumerate(sequence_list):
        if Dataset == "otb" or Dataset == "uav123":
            sequence_path = video['path']
            nz = video['nz']
            ext = video['ext']
            start_frame = video['startFrame']
            end_frame = video['endFrame']

            init_omit = 0
            if 'initOmit' in video:
                init_omit = video['initOmit']

            image_list = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=data_dir,
                                                                                   sequence_path=sequence_path,
                                                                                   frame=frame_num, nz=nz, ext=ext) for
                          frame_num in range(start_frame + init_omit, end_frame + 1)]

            anno_path = '{}/{}'.format(data_dir, video['anno_path'])

            try:
                groundtruth = np.loadtxt(str(anno_path), dtype=np.float64)
            except:
                groundtruth = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)
            result_save_path = os.path.join(base_save_path, video['name'] + '.txt')
            image_dir = image_list[0]
        elif Dataset == "lasot":
            sequence_dir = data_dir + '/' + video + '/img/'
            gt_dir = data_dir + '/' + video + '/groundtruth.txt'
            image_list = os.listdir(sequence_dir)
            image_list.sort()
            image_list = [im for im in image_list if im.endswith("jpg") or im.endswith("jpeg")]
            try:
                groundtruth = np.loadtxt(gt_dir, delimiter=',')
            except:
                groundtruth = np.loadtxt(gt_dir)
            result_save_path = os.path.join(base_save_path, video + '.txt')
            image_dir = sequence_dir + image_list[0]


        if os.path.exists(result_save_path):
            continue

        region = Region(groundtruth[0, 0], groundtruth[0, 1], groundtruth[0, 2], groundtruth[0, 3])

        # image = cv.cvtColor(cv.imread(image_dir), cv.COLOR_BGR2RGB)
        image = Image.open(image_dir).convert('RGB')
        tracker = metric_tracker(image, region, video=video, p=p)#,groundtruth=groundtruth
        num_frames = len(image_list)
        bBoxes = np.zeros((num_frames, 4))
        bBoxes2 = np.zeros((num_frames, 4))

        bBoxes[0, :] = groundtruth[0, :]
        bBoxes2[0, :] = groundtruth[0, :]
        for im_id in range(1, len(image_list)):
            if Dataset=='lasot':
                imagefile = sequence_dir + image_list[im_id]
            else:
                imagefile = image_list[im_id]
            # image = cv.cvtColor(cv.imread(imagefile), cv.COLOR_BGR2RGB)
            image = Image.open(imagefile).convert('RGB')
            # print("%d: " % seq_id + video + ": %d /" % im_id + "%d" % len(image_list))
            if p.base_tracker=='pymdnet':
                region = tracker.pymdnet_track(image)
            elif p.base_tracker=='metricnet':
                region= tracker.metricnet_track(image)
            if p.visualization:
                show_res(cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR), np.array(region).astype('int16'), '1')
            bBoxes[im_id, :] = region
            # bbox_overlap[im_id] = overlap_ratio(groundtruth[im_id], bBoxes[im_id, :])[0]
            #print(region)
        fps=tracker.i/tracker.spf_total
        print('fps',fps)
        if save:
            np.savetxt(result_save_path, bBoxes, fmt="%.6f,%.6f,%.6f,%.6f")
            # np.savetxt(result_save_path2,save_cont,fmt="%.8f,%.8f,%.8f,%.8f,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f")
        # np.savetxt(os.path.join('/home/daikenan/Desktop/MDNet_tf/', video+'.txt'), bBoxes2, fmt="%.6f,%.6f,%.6f,%.6f")


class p_config(object):
    # base_tracker = 'pymdnet'
    dataset='otb'
    base_tracker='metricnet'
    visualization =False
    if base_tracker=='pymdnet':
        save_path = 'results-trackingNet/pymdnet/'
    else:
        save_path = '/media/zj/4T-2/metricnet_otb'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
def main(_):
    # print('in main function')
    p = p_config()

    # videos = ['ballet', 'bicycle', 'boat', 'car1', 'cat2', 'deer', 'volkswagen', 'dog', 'bull', 'car6', 'cat1',
    #           'carchase', 'dragon', 'f1', 'freesbiedog', 'group1', 'horseride', 'longboard', 'parachute', 'sup',
    #           'tightrope', 'uav1', 'volkswagen', 'warmup']
    # videos = ['boat']
    # for i in range(len(videos)):
    #     eval_tracking('votlt19', p=p, video_spe=videos[i])
    #eval_tracking('lasot', p=p, save=True, video_spe='airplane-1')
    if not os.path.exists(p.save_path):
        os.mkdir(p.save_path)
    if p.dataset=='lasot':
        file = open('/media/zj/4T/Dataset/LaSOT/dataset/images/test0.txt', 'r')
        videos = file.readlines()
        for id, video in enumerate(videos):
            video = video.strip('\n')  # airplane-1
            # video='basketball-1'
            print(video)
            eval_tracking('lasot', p=p, video_spe=video, save=True)
    elif p.dataset=='otb':
        videos = get_otb_info_list()
        for id in range(len(videos)):
            video = videos[id]['name']
            print(video)
            eval_tracking('otb', p=p, video_spe=videos[id], save=True)
    elif p.dataset=='uav123':
        videos = get_uav_info_list()
        for id in range(len(videos)):
            video = videos[id]['name']
            print(video)
            eval_tracking('uav123', p=p, video_spe=videos[id], save=True)





if __name__ == '__main__':
    tf.app.run()
