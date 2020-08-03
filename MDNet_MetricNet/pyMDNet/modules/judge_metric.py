import os
import torch
from me_sample_generator import *
from PIL import Image
from pyMDNet.modules.metric_model import ft_net
from torch.autograd import Variable
import time

def get_anchor_feature(model, im, box):
    anchor_region = me_extract_regions(im, box)
    anchor_region = process_regions(anchor_region)
    anchor_region = torch.Tensor(anchor_region)
    anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
    model.eval()
    anchor_feature = model(anchor_region)
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

def Judge( anchor_feature, pos_feature,flag,opts):
    #threshold = 8.0828
    #threshold = 4.0
    pos_feature=pos_feature.repeat(anchor_feature.shape[0],1)
    ap_dist = torch.norm(anchor_feature - pos_feature, 2, dim=1).view(-1)
    del pos_feature
    if flag:#pos
        threshold = opts['pos_thresh']
        #print('pos',ap_dist.min().cpu().detach().numpy(),ap_dist.max().cpu().detach().numpy())
        result=ap_dist<threshold
    else:#neg
        threshold = opts['neg_thresh']#4.0
        #print('neg', ap_dist.min().cpu().detach().numpy(), ap_dist.max().cpu().detach().numpy())
        result=ap_dist>threshold
    return result,ap_dist

def model_load(path):
    model = ft_net(class_num=1120)
    model.eval()
    model = model.cuda()
    model.load_state_dict(torch.load(path),strict=False)
    return model


def judge_metric(model,anchor_box,im,target_feature,flag,opts):#flag=1(pos),flag=0(neg)
    anchor_feature = get_anchor_feature(model, im, anchor_box)  # anchor_box: (1,4) x,y,w,h
    result,ap_dist = Judge(anchor_feature, target_feature,flag,opts)
    result=result.cpu().numpy()
    result.dtype='bool'
    anchor_box=anchor_box[result]
    del anchor_feature
    return anchor_box,result,ap_dist.cpu().detach().numpy()

def judge_metric_lof(model,anchor_box,im,target_feature,clf,opts):#flag=1(pos),flag=0(neg)
    anchor_feature = get_anchor_feature(model, im, anchor_box)  # anchor_box: (1,4) x,y,w,h
    anchor_feature=anchor_feature.cpu().detach().numpy()
    predict = -clf._score_samples(anchor_feature)
    result=predict <= opts['lof_thresh']
    anchor_box=anchor_box[result]
    del anchor_feature
    return anchor_box

def judge_metric_center(model,neg_examples,im,pos_center,opts):
    neg_features=get_anchor_feature(model,im,neg_examples)
    neg_center=neg_features.mean(dim=0).reshape((1,neg_features.shape[1]))
    pos_center=pos_center.repeat(neg_features.shape[0],1)
    neg_center=neg_center.repeat(neg_features.shape[0],1)
    nn_dist=torch.norm(neg_features-neg_center,2,dim=1).view(-1)
    np_dist=torch.norm(neg_features-pos_center,2,dim=1).view(-1)

    nn_dist=nn_dist*opts['neg_center_thresh']
    result=np_dist>nn_dist
    result = result.cpu().numpy()
    result.dtype = 'bool'
    return result

def judge_success(model,anchor_feature,target_feature,opts):#flag=1(pos),flag=0(neg)
    result,ap_dist = Judge(anchor_feature, target_feature,1,opts)
    return result,ap_dist.cpu().detach().numpy()

def get_metric_dist(model,anchor_box,im,target_feature,opts):
    anchor_feature = get_anchor_feature(model, im, anchor_box)  # anchor_box: (1,4) x,y,w,h
    _, dist = Judge(anchor_feature, target_feature, 1, opts)
    del anchor_feature
    return dist
def get_metric_dist_lof(model,anchor_box,im,target_feature,opts):
    anchor_feature = get_anchor_feature(model, im, anchor_box)  # anchor_box: (1,4) x,y,w,h
    _, dist = Judge(anchor_feature, target_feature, 1, opts)
    # del anchor_feature
    return anchor_feature,dist

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
    pos_feature= model(pos_region)#_ is class_result
    #class_result = torch.softmax(class_result, dim=1)
    return pos_feature
