import os
import torch
from pytracking.metricnet.me_sample_generator import *
from PIL import Image
from pytracking.metricnet.metric_model import ft_net_lof_fc,ft_net_lof_cnn,ft_net_lof_fc_bottleneck,ft_net_lof_fc_bottleneck_512,ft_net_lof_fc_bottleneck_128
from torch.autograd import Variable
import numpy as np
import cv2
import time

def get_anchor_feature(model, im, box):
    anchor_region = me_extract_regions(im, box)
    anchor_region = process_regions(anchor_region)
    anchor_region = torch.Tensor(anchor_region)
    anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
    model.eval()
    _,anchor_feature = model(anchor_region)
    del anchor_region
    return anchor_feature

def process_regions(regions):
    # regions = np.squeeze(regions, axis=0)
    regions = regions / 255.0
    regions[:,:, :, 0] = (regions[:,:, :, 0] - 0.485) / 0.229
    regions[:,:, :, 1] = (regions[:,:, :, 1] - 0.456) / 0.224
    regions[:,:, :, 2] = (regions[:,:, :, 2] - 0.406) / 0.225
    regions = np.transpose(regions, (0,3, 1, 2))
    #regions = np.expand_dims(regions, axis=0)
    #regions = np.tile(regions, (2,1,1,1))
    return regions
def process_regions2(regions):
    # regions = np.squeeze(regions, axis=0)
    regions = regions / 255.0
    regions[:, 1, :, :] = (regions[:, 1, :, :] - 0.456) / 0.224
    regions[:, 2, :, :] = (regions[:, 2, :, :] - 0.406) / 0.225
    regions[:, 0, :, :] = (regions[:, 0, :, :] - 0.485) / 0.229
    return regions

def Judge( anchor_feature, pos_feature,flag,opts):
    #threshold = 8.0828
    # threshold = 4.0
    pos_feature=pos_feature.repeat(anchor_feature.shape[0],1)
    ap_dist = torch.norm(anchor_feature - pos_feature, 2, dim=1).view(-1)
    del pos_feature
    if flag==1:#pos
        # threshold = opts['pos_thresh']
        #print('pos',ap_dist.min().cpu().detach().numpy(),ap_dist.max().cpu().detach().numpy())
        threshold = opts.pos_thresh#4.5
        result=ap_dist<threshold
    else:#neg
        # threshold = opts['neg_thresh']#4.0
        #print('neg', ap_dist.min().cpu().detach().numpy(), ap_dist.max().cpu().detach().numpy())
        threshold = 4.0
        result=ap_dist>threshold
    return result,ap_dist

def Judge2( model,anchor_feature, pos_feature,flag,opts):
    #threshold = 8.0828
    # threshold = 4.0
    pos_feature=pos_feature.repeat(anchor_feature.shape[0],1)
    # ap_dist = torch.norm(anchor_feature - pos_feature, 2, dim=1).view(-1)
    ap_metric=abs(anchor_feature-pos_feature)
    score=model.classifier_2(ap_metric)
    del pos_feature,ap_metric
    if flag==1:#pos
        # threshold = opts['pos_thresh']
        #print('pos',ap_dist.min().cpu().detach().numpy(),ap_dist.max().cpu().detach().numpy())
        result=score[:,1]>1
    else:#neg
        # threshold = opts['neg_thresh']#4.0
        #print('neg', ap_dist.min().cpu().detach().numpy(), ap_dist.max().cpu().detach().numpy())
        # threshold = 4.0
        result=score[:,1]<1
    return result,score[:,1]

def Judge_classifier( model,anchor_feature, pos_feature,flag,opts):
    #threshold = 8.0828
    #threshold = 4.0
    pos_feature=pos_feature.repeat(anchor_feature.shape[0],1)
    ap_metric=abs(anchor_feature-pos_feature)
    score=model.classifier_2(ap_metric)
    score=torch.softmax(score,dim=1)
    # ap_dist = torch.norm(anchor_feature - pos_feature, 2, dim=1).view(-1)
    del pos_feature
    if flag==1:#pos
        result=score[:,1]>score[:,0]
    else:#neg
        result=score[:,1]<score[:,0]
    return result,score[:,1]

def model_load(path):
    model = ft_net_lof_fc(class_num=1120)
    model.eval()
    model = model.cuda()
    model.load_state_dict(torch.load(path),strict=False)
    return model
def model_load_bottleneck(path):
    model = ft_net_lof_fc_bottleneck(class_num=1120)
    model.eval()
    model = model.cuda()
    model.load_state_dict(torch.load(path),strict=False)
    return model
def model_load_bottleneck_512(path):
    model = ft_net_lof_fc_bottleneck_512(class_num=1120)
    model.eval()
    model = model.cuda()
    model.load_state_dict(torch.load(path),strict=False)
    return model
def model_load_bottleneck_128(path):
    model = ft_net_lof_fc_bottleneck_128(class_num=1120)
    model.eval()
    model = model.cuda()
    model.load_state_dict(torch.load(path),strict=False)
    return model
def model_load_cnn(path):
    model = ft_net_lof_cnn(class_num=1120)
    model.eval()
    model = model.cuda()
    model.load_state_dict(torch.load(path),strict=False)
    return model

def show_feature(model,anchor_box,im):
    anchor_region = me_extract_regions(im, anchor_box)
    anchor_region = process_regions(anchor_region)
    anchor_region = torch.Tensor(anchor_region)
    anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
    model.eval()
    anchor_feature = model(anchor_region,'roi')
    anchor_mean = anchor_feature.mean(1)
    anchor_mean = anchor_mean.cpu().data.numpy()
    anchor_mean = np.round(anchor_mean * 255)
    for i in range(anchor_mean.shape[0]):
        feature = np.round(anchor_mean[i])
        cv2.imwrite('/home/zj/tracking/metricNet/tmp_fig/' + str(i) + '_v0.jpg', feature)

def show_feature_roi_align(anchor_feature):
    anchor_mean = anchor_feature.mean(1)
    anchor_mean = anchor_mean.cpu().data.numpy()
    anchor_mean = np.round(anchor_mean * 255)
    for i in range(anchor_mean.shape[0]):
        feature = np.round(anchor_mean[i])
        cv2.imwrite('/home/zj/tracking/metricNet/tmp_fig/' + str(i) + '_roi.jpg', feature)

def judge_metric(model,anchor_box,im,target_feature,flag,opts):#flag=1(pos),flag=0(neg)
    anchor_feature = get_anchor_feature(model, im, anchor_box)  # anchor_box: (1,4) x,y,w,h
    result,_ = Judge(anchor_feature, target_feature,flag,opts)
    result=result.cpu().numpy()
    result.dtype='bool'
    anchor_box=anchor_box[result]
    del anchor_feature
    return anchor_box,result

def judge_metric2(model,anchor_box,anchor_feature_tmp,target_feature,flag,opts):#flag=1(pos),flag=0(neg)
    model.eval()
    # print('shape:',anchor_box.shape[0])
    # tic=time.time()
    anchor_feature = model.forward_tmp(anchor_feature_tmp)  # anchor_box: (1,4) x,y,w,h
    # print('layer4:',time.time()-tic)
    # tic=time.time()
    # anchor_feature = model.forward_tmp2(anchor_feature)
    # print('distance:',time.time()-tic)
    # target_feature=model.forward_tmp(target_feature)
    result,dist = Judge(anchor_feature, target_feature,flag,opts)
    result=result.cpu().numpy()
    result.dtype='bool'
    anchor_box=anchor_box[result]
    del anchor_feature
    return anchor_box,result,dist
def judge_metric2_classifier(model,anchor_box,anchor_feature_tmp,target_feature,flag,opts):#flag=1(pos),flag=0(neg)
    model.eval()
    # print('shape:',anchor_box.shape[0])
    # tic=time.time()
    anchor_feature = model.forward_tmp(anchor_feature_tmp)  # anchor_box: (1,4) x,y,w,h
    # print('layer4:',time.time()-tic)
    # tic=time.time()
    # anchor_feature = model.forward_tmp2(anchor_feature)
    # print('distance:',time.time()-tic)
    # target_feature=model.forward_tmp(target_feature)
    result,_= Judge_classifier(model,anchor_feature, target_feature,flag,opts)
    result=result.cpu().numpy()
    result.dtype='bool'
    anchor_box=anchor_box[result]
    del anchor_feature
    return anchor_box,result
def judge_success(model,anchor_feature,target_feature,opts):#flag=1(pos),flag=0(neg)
    result,ap_dist = Judge(anchor_feature, target_feature,1,opts)
    return result,ap_dist.cpu().detach().numpy()

def judge_success_no_class(model,anchor_feature,target_feature,opts):#flag=1(pos),flag=0(neg)
    # result,ap_dist = Judge(anchor_feature, target_feature,1,opts)
    result, ap_dist = Judge2(model,anchor_feature, target_feature, 1, opts)
    return result,ap_dist.cpu().detach().numpy()

def judge_success_classifier(model,anchor_feature,target_feature,opts):#flag=1(pos),flag=0(neg)
    result,ap_score= Judge_classifier(model,anchor_feature, target_feature,1,opts)
    return result,ap_score.cpu().detach().numpy()


def get_metric_dist(model,anchor_box,im,target_feature,opts):
    anchor_feature = get_anchor_feature(model, im, anchor_box)  # anchor_box: (1,4) x,y,w,h
    _, dist = Judge(anchor_feature, target_feature, 1, opts)
    del anchor_feature
    return dist

def get_metric_dist2(model,anchor_feature_tmp,target_feature,opts):
    model.eval()
    anchor_feature = model.forward_tmp(anchor_feature_tmp)
    _, dist = Judge(anchor_feature, target_feature, 1, opts)
    del anchor_feature
    return dist

def get_metric_dist_by_feature(model,target_feature_all,current_feature,opts):
    dist=np.zeros(len(target_feature_all))
    for i in range(len(target_feature_all)):
        _, dist[i] = Judge(target_feature_all[i], current_feature, 1, opts)
    return dist

def get_target_feature(model,pos_box,im):
    pos_box = pos_box.reshape((1, 4))
    pos_region = me_extract_regions(im, pos_box)
    pos_region = process_regions(pos_region)
    pos_region = torch.Tensor(pos_region)
    pos_region = (Variable(pos_region)).type(torch.FloatTensor).cuda()
    model.eval()
    _,pos_feature= model(pos_region)#_ is class_result
    #class_result = torch.softmax(class_result, dim=1)
    return pos_feature