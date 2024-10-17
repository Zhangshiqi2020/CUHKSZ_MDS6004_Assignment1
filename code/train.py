import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn

import os
import time
import shutil
import argparse

import numpy as np
from yaml import load, FullLoader
from easydict import EasyDict as edict
from utils import Logger, FocalLoss, KDLoss

from dataset import myDataSet


def get_parameters(model, args, useDeconvGroup=True):
    lr_0 = []
    lr_1 = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if 'deconv' in key and useDeconvGroup==True:
            print ("useDeconvGroup=True, lr=0, key: ", key)
            lr_0.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_0, 'lr': args.lr * 0},
              {'params': lr_1, 'lr': args.lr * 1}]
    return params, [0., 1.]

def adjust_learning_rate(optimizer, epoch, args, multiple):
    """Sets the learning rate to the initial LR decayed by 0.95 every 20 epochs"""
    # lr = args.lr * (0.95 ** (epoch // 4))
    lr = args.lr * (0.95 ** (epoch // 20))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    pass

def save_checkpoint(state, is_best, root, filename='checkpoint.pth.tar'):
    torch.save(state, root+filename)
    if is_best:
        shutil.copyfile(root+filename, root+'model_best.pth.tar')

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    pass

def train(dataLoader, netmodel, optimizer, epoch, logger, exp_args):
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    
    losses = AverageMeter('losses')
    losses_mask = AverageMeter('losses_mask')
    
    losses_edge_ori = AverageMeter('losses_edge_ori')
    losses_edge = AverageMeter('losses_edge')
    
    losses_mask_ori = AverageMeter('losses_mask_ori')
    losses_stability_mask = AverageMeter('losses_stability_mask')
    losses_stability_edge = AverageMeter('losses_stability_edge')

    netmodel.train()  # Switch to train mode
    
    loss_Softmax = nn.CrossEntropyLoss(ignore_index=255)  # Mask loss
    loss_Focalloss = FocalLoss(gamma=2)  # Boundary loss
    loss_KL = KDLoss() # knowledge-distillation (KD) loss

    
    scaler = amp.GradScaler()  # Initialize GradScaler for mixed precision
    
    end = time.time()
    
    for i, (input_ori, input, edge, mask) in enumerate(dataLoader):  
        data_time.update(time.time() - end)
        
        input_ori_var = input_ori.cuda()
        input_var = input.cuda()
        edge_var = edge.cuda()
        mask_var = mask.cuda()

        optimizer.zero_grad()  # Zero the gradients
        
        with amp.autocast():  # Automatic mixed precision enabled
            # loss_mask + loss_edge
            output_mask, output_edge = netmodel(input_var)
            loss_mask = loss_Softmax(output_mask, mask_var)
            losses_mask.update(loss_mask.item(), input.size(0))

            loss_edge = loss_Focalloss(output_edge, edge_var) * exp_args.edgeRatio
            losses_edge.update(loss_edge.item(), input.size(0))
            
            # loss_stability_mask + loss_edge_ori
            output_mask_ori, output_edge_ori = netmodel(input_ori_var)

            loss_mask_ori = loss_Softmax(output_mask_ori, mask_var)
            losses_mask_ori.update(loss_mask_ori.item(), input.size(0))
            
            loss_edge_ori = loss_Focalloss(output_edge_ori, edge_var) * exp_args.edgeRatio
            losses_edge_ori.update(loss_edge_ori.item(), input.size(0))
            
            loss_stability_mask = loss_KL(output_mask, output_mask_ori.detach(), exp_args.temperature) * exp_args.alpha
            losses_stability_mask.update(loss_stability_mask.item(), input.size(0))

            loss_stability_edge = loss_KL(output_edge, output_edge_ori.detach(), exp_args.temperature) * exp_args.alpha * exp_args.edgeRatio
            losses_stability_edge.update(loss_stability_edge.item(), input.size(0))
            
            loss = loss_mask + loss_edge + loss_stability_mask + loss_edge_ori

        # Update total loss
        losses.update(loss.item(), input.size(0))

        # Perform the backward pass using the scaled loss
        scaler.scale(loss).backward()

        # Update the optimizer step with the scaled gradients
        scaler.step(optimizer)

        # Update the scaler for the next iteration
        scaler.update()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Logging and printing (omitted for brevity)
        if i % args.printfreq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(dataLoader)}]\t'
                  f'Lr-deconv: [{optimizer.param_groups[0]["lr"]}]\t'
                  f'Lr-other: [{optimizer.param_groups[1]["lr"]}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')

        ## '===========> logger <==========='
        # (1) Log the scalar values
        info = {losses.name: losses.val,
                # losses_mask_ori.name: losses_mask_ori.val,
                # losses_mask.name: losses_mask.val,
                # losses_edge_ori.name: losses_edge_ori.val,
                # losses_edge.name: losses_edge.val,
                # losses_stability_mask.name: losses_stability_mask.val, 
                # losses_stability_edge.name: losses_stability_edge.val
                }
        for tag, value in info.items():
            logger.scalar_summary(tag = tag, value = value, step = i)
    pass

def test(dataLoader, netmodel, optimizer, epoch, logger, exp_args):

    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')

    losses = AverageMeter('losses')
    
    losses_mask_ori = AverageMeter('losses_mask_ori')
    losses_mask = AverageMeter('losses_mask')
    
    losses_edge_ori = AverageMeter('losses_edge_ori')
    losses_edge = AverageMeter('losses_edge')
    
    losses_stability_mask = AverageMeter('losses_stability_mask')
    losses_stability_edge = AverageMeter('losses_stability_edge')
    
    # switch to eval mode
    netmodel.eval()
    
    loss_Softmax = nn.CrossEntropyLoss(ignore_index=255)  # Mask loss
    loss_Focalloss = FocalLoss(gamma=2)  # Edge loss
    loss_KL = KDLoss()  # KD loss

    end = time.time()
    softmax = nn.Softmax(dim=1)
    iou = 0
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for i, (input_ori, input, edge, mask) in enumerate(dataLoader):  
            data_time.update(time.time() - end)

            # Move inputs and labels to GPU
            input_ori_var = input_ori.cuda()
            input_var = input.cuda()
            edge_var = edge.cuda()
            mask_var = mask.cuda()

            with amp.autocast():  # Enable automatic mixed precision
                output_mask, output_edge = netmodel(input_var)

                loss_mask = loss_Softmax(output_mask, mask_var)
                losses_mask.update(loss_mask.item(), input.size(0))

                loss_edge = loss_Focalloss(output_edge, edge_var) * exp_args.edgeRatio
                losses_edge.update(loss_edge.item(), input.size(0))


                output_mask_ori, output_edge_ori = netmodel(input_ori_var)

                loss_mask_ori = loss_Softmax(output_mask_ori, mask_var)
                losses_mask_ori.update(loss_mask_ori.item(), input.size(0))

                loss_edge_ori = loss_Focalloss(output_edge_ori, edge_var) * exp_args.edgeRatio
                losses_edge_ori.update(loss_edge_ori.item(), input.size(0))

                loss_stability_mask = loss_KL(output_mask, output_mask_ori.detach(), exp_args.temperature) * exp_args.alpha
                losses_stability_mask.update(loss_stability_mask.item(), input.size(0))

                loss_stability_edge = loss_KL(output_edge, output_edge_ori.detach(), exp_args.temperature) * exp_args.alpha * exp_args.edgeRatio
                losses_stability_edge.update(loss_stability_edge.item(), input.size(0))

                loss = loss_mask + loss_edge + loss_stability_mask + loss_stability_edge

            # total loss
            loss = loss_mask
            losses.update(loss.item(), input.size(0))

            # Prediction and IOU calculation
            prob = softmax(output_mask)[0, 1, :, :]
            pred = prob.data.cpu().numpy()
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            iou += calcIOU(pred, mask_var[0].data.cpu().numpy())

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print and log progress
            if i % args.printfreq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(dataLoader)}]\t'
                      f'Lr-deconv: [{optimizer.param_groups[0]["lr"]}]\t'
                      f'Lr-other: [{optimizer.param_groups[1]["lr"]}]\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})')

        ## '===========> logger <==========='
        # (1) Log the scalar values
        info = {losses.name: losses.val,
                # losses_mask_ori.name: losses_mask_ori.val,
                # losses_mask.name: losses_mask.val,
                # losses_edge_ori.name: losses_edge_ori.val,
                # losses_edge.name: losses_edge.val,
                # losses_stability_mask.name: losses_stability_mask.val, 
                # losses_stability_edge.name: losses_stability_edge.val
                }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step=i)
    return 1-iou/len(dataLoader)



def main(args):
    cudnn.benchmark = True
    assert args.model in ['PortraitNet', 'ENet', 'BiSeNet'], 'Error!, <model> should in [PortraitNet, ENet, BiSeNet]'
    
    config_path = args.config_path
    print ('===========> loading config <============')
    print ("config path: ", config_path)
    with open(config_path,'rb') as f:
        cont = f.read()
    cf = load(cont,Loader = FullLoader)
    
    print ('===========> loading data <===========')
    exp_args = edict()
    
    exp_args.istrain = cf['istrain'] # set the mode 
    exp_args.task = cf['task'] # only support 'seg' now
    exp_args.datasetlist = cf['datasetlist']
    exp_args.model_root = cf['model_root'] 
    exp_args.data_root = cf['data_root']
    exp_args.file_root = cf['file_root']
    # whether to continue training
    exp_args.resume = cf['resume'] 
    
    # set log path
    # logs_path = os.path.join(exp_args.model_root, 'log/')
    # if exp_args.resume != True:
    #     os.path.exists(logs_path)
    #     shutil.rmtree(logs_path)


    logs_path = os.path.join(exp_args.model_root, 'log/')

    # 如果 resume 为 False，删除旧的 log 目录并重新创建
    if exp_args.resume != True:
        if os.path.exists(logs_path):
            shutil.rmtree(logs_path)
        os.makedirs(logs_path, exist_ok=True)
    else:
        # 如果日志目录不存在，则创建它
        if not os.path.exists(logs_path):
            os.makedirs(logs_path, exist_ok=True)


    # set log path (train/test)
    logger_train = Logger(logs_path + 'train')
    logger_test = Logger(logs_path + 'test')
    
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
    # the weight of boundary auxiliary loss, default=0.1
    exp_args.edgeRatio = cf['edgeRatio']
    # whether to add consistency constraint loss, default=False
    exp_args.stability = cf['stability']
    # whether to use KL loss in consistency constraint loss, default=True
    exp_args.use_kl = cf['use_kl']
    # temperature in consistency constraint loss, default=1
    exp_args.temperature = cf['temperature'] 
    # the weight of consistency constraint loss, default=2
    exp_args.alpha = cf['alpha'] 
    
    # input normalization parameters
    exp_args.padding_color = cf['padding_color']
    exp_args.img_scale = cf['img_scale']
    # BGR order, image mean, default=[103.94, 116.78, 123.68]
    exp_args.img_mean = cf['img_mean']
    # BGR order, image val, default=[1/0.017, 1/0.017, 1/0.017]
    exp_args.img_val = cf['img_val'] 
    
    # whether to use pretian model to init portraitnet
    exp_args.init = cf['init'] 

    
    # if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
    exp_args.useUpsample = cf['useUpsample'] 
    # if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
    exp_args.useDeconvGroup = cf['useDeconvGroup'] 

    exp_args.faceDetection = cf['faceDetection']
    exp_args.offSet = cf['offSet']
    
    # set training dataset
    exp_args.istrain = True
    dataset_train = myDataSet(exp_args)
    print ("image number in training: ", len(dataset_train))
    dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, 
                                                   shuffle=True, num_workers= args.workers)
    
    # set testing dataset
    exp_args.istrain = False
    dataset_test = myDataSet(exp_args)
    print ("image number in testing: ", len(dataset_test))
    dataLoader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, 
                                                  shuffle=False, num_workers=args.workers)
    
    exp_args.istrain = True
    print ("finish load dataset ...")
    
    print ('===========> loading model <===========')
    
    if args.model == 'PortraitNet':
        # train our model: portraitnet
        import portraitnet
        netmodel = portraitnet.MobileNetV2(n_class=2, 
                                        useUpsample=exp_args.useUpsample, 
                                        useDeconvGroup=exp_args.useDeconvGroup, 
                                        addEdge=exp_args.addEdge, 
                                        channelRatio=1.0, 
                                        minChannel=16, 
                                        weightInit=True,
                                        video=exp_args.video).cuda()
        print ("finish load PortraitNet ...")

    params, multiple = get_parameters(netmodel, args, useDeconvGroup=exp_args.useDeconvGroup)

    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weightdecay) 
    
    if exp_args.resume:
        bestModelFile = os.path.join(exp_args.model_root, 'model_best.pth.tar')
        if os.path.isfile(bestModelFile):
            checkpoint = torch.load(bestModelFile)
            netmodel.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            gap = checkpoint['epoch']
            minLoss = checkpoint['minLoss']
            print("=> loaded checkpoint '{}' (epoch {})".format(bestModelFile, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(bestModelFile))
    else:
        minLoss = 10000
        gap = 0
        
    for epoch in range(gap, 200):
        adjust_learning_rate(optimizer, epoch, args, multiple)
        print ('===========>   training    <===========')
        train(dataLoader_train, netmodel, optimizer, epoch, logger_train, exp_args)
        print ('===========>   testing    <===========')
        loss = test(dataLoader_test, netmodel, optimizer, epoch, logger_test, exp_args)
        print ("loss: ", loss, minLoss)
        is_best = False
        if loss < minLoss:
            minLoss = loss
            is_best = True
        
        save_checkpoint({
            'epoch': epoch+1,
            'minLoss': minLoss,
            'state_dict': netmodel.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, is_best, exp_args.model_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training code')
    parser.add_argument('--model', default='PortraitNet', type=str, 
                        help='<model> should in [PortraitNet, ENet, BiSeNet]')
    # EG1800 Dataset
    parser.add_argument('--config_path', 
                        default='D:\Code\MDS_6004\My_model\config_EG1800.yaml', 
                        type=str, help='the config path of the model')

    # EG1800_face_detection Dataset
    # parser.add_argument('--config_path', 
    #                 default='D:\Code\MDS_6004\My_model\config_EG1800_face_detection.yaml', 
    #                 type=str, help='the config path of the model')
    
    # MattingHuman Dataset
    # parser.add_argument('--config_path', 
    #                 default='D:\Code\MDS_6004\My_model\config_mattinghuman.yaml', 
    #                 type=str, help='the config path of the model')

    # EasyPortrait_eyes Dataset
    # parser.add_argument('--config_path', 
    #                 default='D:\Code\MDS_6004\My_model\config_EasyPortrait_eyes.yaml', 
    #                 type=str, help='the config path of the model')

    # EasyPortrait_teeth Dataset
    # parser.add_argument('--config_path', 
    #                 default='D:\Code\MDS_6004\My_model\config_EasyPortrait_teeth.yaml', 
    #                 type=str, help='the config path of the model')
    
    parser.add_argument('--workers', default=3, type=int, help='number of data loading workers')
    parser.add_argument('--batchsize', default=8, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--printfreq', default=100, type=int, help='print frequency')
    parser.add_argument('--savefreq', default=1000, type=int, help='save frequency')
    parser.add_argument('--resume', default=False, type=bool, help='resume')
    args = parser.parse_args()
    
    main(args)