import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel

import os
import cv2
import numpy as np
from easydict import EasyDict as edict
import yaml

import sys
import dataset
from dataset import myDataSet
from data_aug import Normalize_Img, Anti_Normalize_Img

import matplotlib
import matplotlib.pyplot as plt
import pylab

def calcIOU(img, mask):
    sum1 = img + mask
    sum1[sum1>0] = 1
    sum2 = img + mask
    sum2[sum2<2] = 0
    sum2[sum2>=2] = 1
    if np.sum(sum1) == 0:
        return 1
    else:
        return 1.0*np.sum(sum2)/np.sum(sum1)

def test(dataLoader, netmodel, exp_args):
    # switch to eval mode
    netmodel.eval()
    softmax = nn.Softmax(dim=1)
    iou = 0
    for i, (input_ori, input, edge, mask) in enumerate(dataLoader):  
        input_ori_var = Variable(input_ori.cuda())
        input_var = Variable(input.cuda())
        edge_var = Variable(edge.cuda())
        mask_var = Variable(mask.cuda())
        
        # compute output: loss part1
        if exp_args.addEdge == True:
            output_mask, output_edge = netmodel(input_ori_var)
        else:
            output_mask = netmodel(input_ori_var)
            
        prob = softmax(output_mask)[0,1,:,:]
        pred = prob.data.cpu().numpy()
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0
        iou += calcIOU(pred, mask_var[0].data.cpu().numpy())
        
    print(len(dataLoader))
    return iou/len(dataLoader)


#################################################################
# Set the configuration path here
#################################################################

# EG1800: config_path = 'config_EG1800.yaml'
# config_path = 'config_EG1800.yaml'

# EG1800_2000Epoch: config_path = 'config_EG1800_2000Epoch.yaml'
config_path = 'config_EG1800_2000Epoch.yaml'

# EG1800_face_detection: config_path = 'config_EG1800_face_detection.yaml'
# config_path = 'config_EG1800_face_detection.yaml'

# mattinghuman: config_path = 'config_mattinghuman.yaml'
# config_path = 'config_mattinghuman.yaml'

# mattinghuman: config_path = 'config_mattinghuman_test_on_eg1800.yaml'
# config_path = 'config_mattinghuman_test_on_eg1800.yaml'

# EasyPortraint_eyes: config_path = 'config_EasyPortrait_eyes.yaml'
# Remenber to change the n_class from 2 to 3
# config_path = 'config_EasyPortrait_eyes.yaml'

# EasyPortraint_teeth: config_path = 'config_EasyPortrait_teeth.yaml'
# Remenber to change the n_class from 3 to 2
# config_path = 'config_EasyPortrait_teeth.yaml'


#################################################################
# Set the configuration
#################################################################

with open(config_path,'rb') as f:
    cont = f.read()
cf = yaml.safe_load(cont)

print ('finish load config file ...')

print ('===========> loading data <===========')
exp_args = edict()    
exp_args.istrain = False
exp_args.task = cf['task']
exp_args.datasetlist = cf['datasetlist'] # [EG1800, mattinghuman, EasyPortrait]
print ("datasetlist: ", exp_args.datasetlist)

exp_args.model_root = cf['model_root'] 
exp_args.data_root = cf['data_root']
exp_args.file_root = cf['file_root']

# the height of input images, default=224
exp_args.input_height = cf['input_height']
# the width of input images, default=224
exp_args.input_width = cf['input_width']

# if exp_args.video=True, add prior channel for input images, default=False
exp_args.video = cf['video']
# the probability to set empty prior channel, default=0.5
exp_args.prior_prob = cf['prior_prob']

# whether to add boundary auxiliary loss, default=False
exp_args.addEdge = cf['addEdge']
# whether to add consistency constraint loss, default=False
exp_args.stability = cf['stability']

# input normalization parameters
exp_args.padding_color = cf['padding_color']
exp_args.img_scale = cf['img_scale']
# BGR order, image mean, default=[103.94, 116.78, 123.68]
exp_args.img_mean = cf['img_mean']
# BGR order, image val, default=[1/0.017, 1/0.017, 1/0.017]
exp_args.img_val = cf['img_val'] 

# if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
exp_args.useUpsample = cf['useUpsample'] 
# if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
exp_args.useDeconvGroup = cf['useDeconvGroup']
exp_args.faceDetection = cf['faceDetection']
exp_args.offSet = cf['offSet']

exp_args.init = False
exp_args.resume = True

dataset_test = myDataSet(exp_args)
print(len(dataset_test))
dataLoader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=1)
print(len(dataLoader_test))
print ("finish load dataset ...")

print ('===========> loading model <===========')
import portraitnet
netmodel = portraitnet.MobileNetV2(n_class=2, 
                                useUpsample=exp_args.useUpsample, 
                                useDeconvGroup=exp_args.useDeconvGroup, 
                                addEdge=exp_args.addEdge, 
                                channelRatio=1.0, 
                                minChannel=16, 
                                weightInit=True,
                                video=exp_args.video)

if exp_args.resume:
    bestModelFile = os.path.join(exp_args.model_root, 'model_best.pth.tar')
    if os.path.isfile(bestModelFile):
        checkpoint = torch.load(bestModelFile, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        netmodel.load_state_dict(checkpoint['state_dict'])
        print("minLoss: ", checkpoint['minLoss'], checkpoint['epoch'])
        print("=> loaded checkpoint '{}' (epoch {})".format(bestModelFile, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(bestModelFile))


if torch.cuda.is_available():
    netmodel = netmodel.cuda()
    cudnn.benchmark = True


if __name__ == "__main__":
    acc = test(dataLoader_test, netmodel, exp_args)
    print ("mean iou: ", acc)
