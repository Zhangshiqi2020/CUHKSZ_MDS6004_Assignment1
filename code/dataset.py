import os
import cv2
import copy
import numpy as np

import torch
import torch.utils.data as data

from data_aug import aug_matrix, padding, Normalize_Img
from data_aug import data_aug_flip, data_aug_color, data_aug_blur
from data_aug import data_aug_noise, show_edge

from PIL import Image

class myDataSet(data.Dataset): 
    def __init__(self, exp_args):
        assert exp_args.task in ['seg'], 'Error!, <task> should in [seg]'
        
        self.exp_args = exp_args
        self.task = exp_args.task
        self.datasetlist = exp_args.datasetlist
        self.data_root = exp_args.data_root # data_root = 'D:/ML_dataset/'
        self.file_root = exp_args.file_root # file_root = 'D:/Code/MDS_6004/PortraitNet-master/data/select_data/'
        
        self.datasets = {}
        self.imagelist = []
        
        # load dataset    
        if 'EG1800' in self.datasetlist:
            ImageRoot = self.data_root + 'EG1800/Images/'
            AnnoRoot = self.data_root + 'EG1800/Labels/'
            ImgIds_Train = self.file_root + 'eg1800_train.txt'
            ImgIds_Test = self.file_root + 'eg1800_test.txt'
            exp_args.dataset = 'eg1800'
            self.datasets['eg1800'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)
        
        ##### the dictionary of the dataset should be changed.#####
        elif 'mattinghuman' in self.datasetlist:
            ImageRoot = self.data_root + 'matting_human_sample/clip_img/'
            AnnoRoot = self.data_root + 'matting_human_sample/matting/'
            ImgIds_Train = self.file_root + 'mattinghuman_train.txt'
            ImgIds_Test = self.file_root + 'mattinghuman_test.txt'
            exp_args.dataset = 'mattinghuman'
            self.datasets['mattinghuman'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)
        
        elif 'EasyPortrait' in self.datasetlist:
            ImageRoot = self.data_root + 'new_archive/images/'
            AnnoRoot = self.data_root + 'new_archive/annotations/'
            ImgIds_Train = self.file_root + 'EasyPortrait_train.txt'
            ImgIds_Test = self.file_root + 'EasyPortrait_test.txt'
            exp_args.dataset = 'EasyPortrait'
            self.datasets['EasyPortrait'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)
        
        else:
            print('Error! <datasetlist> is not correct!')
            exit()

        # image list
        for key in self.datasets.keys():
            length = len(self.datasets[key])
            # print('Dataset: %s, %d images' % (key, length))
            for i in range(length):
                self.imagelist.append([key, i])
        
        # print(self.imagelist)
        
    def __getitem__(self, index):
        subset, subsetidx = self.imagelist[index]
        
        if self.task == 'seg':
            input_ori, input, output_edge, output_mask = self.datasets[subset][subsetidx]
            return input_ori.astype(np.float32), input.astype(np.float32), \
        output_edge.astype(np.int64), output_mask.astype(np.int64)
           
    def __len__(self):
        return len(self.imagelist)
    
class PortraitSeg(data.Dataset): 
    def __init__(self, ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, exp_args):
        self.ImageRoot = ImageRoot
        self.AnnoRoot = AnnoRoot
        self.istrain = exp_args.istrain
        self.stability = exp_args.stability
        self.addEdge = exp_args.addEdge
        
        self.video = exp_args.video
        self.prior_prob = exp_args.prior_prob
        
        self.task = exp_args.task
        self.dataset = exp_args.dataset
        self.input_height = exp_args.input_height
        self.input_width = exp_args.input_width
        
        self.padding_color = exp_args.padding_color
        self.img_scale = exp_args.img_scale
        self.img_mean = exp_args.img_mean # BGR order
        self.img_val = exp_args.img_val # BGR order
        self.faceDetection = exp_args.faceDetection
        self.offSet = exp_args.offSet
        
        if self.istrain == True:
            file_object = open(ImgIds_Train, 'r')
        elif self.istrain == False:
            file_object = open(ImgIds_Test, 'r')
            
        try:
            self.imgIds = file_object.readlines()      
        finally:
             file_object.close()
        pass
            
        
    def __getitem__(self, index):
        '''
        An item is an image. Which may contains more than one person.
        '''
        img = None
        mask = None
        bbox = None
        H = None
        
        if self.dataset in ["eg1800"]:
            # basic info
            img_id = self.imgIds[index].strip()
            img_path = os.path.join(self.ImageRoot, img_id)
            img = cv2.imread(img_path)
            img_name = img_path[img_path.rfind('/')+1:]
            
            # load mask
            annopath = os.path.join(self.AnnoRoot, img_id.replace('.jpg', '.png'))
            mask = cv2.imread(annopath, 0)
            mask[mask>1] = 0
            
            if self.faceDetection == True:
                # Detect faces
                face_cascade = cv2.CascadeClassifier('D:\Code\MDS_6004\My_model\haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                # If face is detected, process the face region
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]

                    x = max(0, x - self.offSet)
                    y = max(0, y - self.offSet)
                    w = min(w + 2 * self.offSet, img.shape[1] - x)
                    h = min(h + 2 * self.offSet, img.shape[0] - y)
                    # Crop the face region from image and mask
                    face_img = img[y:y + h, x:x + w]
                    face_mask = mask[y:y + h, x:x + w]

                    # Resize the face region to model's input size
                    face_img_resized = cv2.resize(face_img, (self.input_width, self.input_height))
                    face_mask_resized = cv2.resize(face_mask, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)

                    # Draw rectangle around the face
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    img = face_img_resized
                    mask = face_mask_resized
                else:
                    # If no face is detected, resize the whole image
                    img = img
                    mask = mask                    

            height, width, channel = img.shape
            bbox = [0, 0, width-1, height-1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                       angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height/4)
            
        elif self.dataset in ["mattinghuman"]:
                        # basic info
            img_id = self.imgIds[index].strip()
            img_path = os.path.join(self.ImageRoot, img_id)
            img = cv2.imread(img_path)
            img_name = img_path[img_path.rfind('/')+1:]
            
            # load mask
            annopath = os.path.join(self.AnnoRoot, img_id.replace('clip_', 'matting_').replace('.jpg', '.png'))
            mask = cv2.imread(annopath, 0)
            mask[mask>1] = 1
            
            height, width, channel = img.shape
            bbox = [0, 0, width-1, height-1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                       angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height/4)
            
            # train_model = ImageRoot: D:/ML_dataset/ new_archive/images/ train/001566b6-b893-4c47-aca0-ecd76f0070c7.jpg
            # test_model = ImageRoot: D:/ML_dataset/ new_archive/images/ test/001566b6-b893-4c47-aca0-ecd76f0070c7.jpg
        
        elif self.dataset in ["EasyPortrait"]:
                        # basic info
            img_id = self.imgIds[index].strip()
            img_path = os.path.join(self.ImageRoot, img_id)
            img = cv2.imread(img_path)
            img_name = img_path[img_path.rfind('/')+1:]
            
            # load mask
            annopath = os.path.join(self.AnnoRoot, img_id.replace('images', 'annotations').replace('.jpg', '.png'))
            mask = cv2.imread(annopath, 0)
            
            ########################################################
            # choose which mask to use
            # ('background', 'skin', 'left brow', 'right brow', 'left eye', 'right eye', 'lips', 'teeth')
            
            # teeth part OK result have
            mask[mask != 7 ] = 0

            # eyes part OK
            # mask[mask > 6] = 0
            # mask[mask < 5] = 0

            # skin park OK
            # mask[mask > 1] = 0

            # left brow part OK
            # mask[mask > 2] = 0
            # mask[mask < 2] = 0
            ########################################################

            height, width, channel = img.shape
            bbox = [0, 0, width-1, height-1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                       angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height/4)
        
        else:
            print('Error! <dataset> is not correct!')
            exit()
              
        use_float_mask = False # use original 0/1 mask as groundtruth
        
        # data augument: first align center to center of dst size. then rotate and scale
        if self.istrain == False:
            img_aug_ori, mask_aug_ori = padding(img, mask, size=self.input_width, padding_color=self.padding_color)
            
            input_norm = Normalize_Img(img_aug_ori, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
            input = np.transpose(input_norm, (2, 0, 1))
            input_ori = copy.deepcopy(input)
        else:
            img_aug = cv2.warpAffine(np.uint8(img), H, (self.input_width, self.input_height), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(self.padding_color, self.padding_color, self.padding_color)) 
            mask_aug = cv2.warpAffine(np.uint8(mask), H, (self.input_width, self.input_height), 
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            img_aug_ori, mask_aug_ori, aug_flag = data_aug_flip(img_aug, mask_aug)
            prior = np.zeros((self.input_height, self.input_width, 1))
                         
            # add augmentation
            img_aug = Image.fromarray(cv2.cvtColor(img_aug_ori, cv2.COLOR_BGR2RGB))  
            img_aug = data_aug_color(img_aug)
            img_aug = np.asarray(img_aug)
            # img_aug = data_aug_light(img_aug)
            img_aug = data_aug_blur(img_aug)
            img_aug = data_aug_noise(img_aug)
            img_aug = np.float32(img_aug[:,:,::-1]) # BGR, like cv2.imread
            
            input_norm = Normalize_Img(img_aug, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
            input_ori_norm = Normalize_Img(img_aug_ori, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
                  
            input = np.transpose(input_norm, (2, 0, 1))
            input_ori = np.transpose(input_ori_norm, (2, 0, 1))
            
        if 'seg' in self.task:
            if use_float_mask == True:
                output_mask = cv2.resize(mask_aug_ori, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
                cv2.normalize(output_mask, output_mask, 0, 1, cv2.NORM_MINMAX)
                output_mask[output_mask>=0.5] = 1
                output_mask[output_mask<0.5] = 0
            else:
                output_mask = cv2.resize(np.uint8(mask_aug_ori), (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
                
                # add mask blur
                output_mask = np.uint8(cv2.blur(output_mask, (5,5)))
                output_mask[output_mask>=0.5] = 1
                output_mask[output_mask<0.5] = 0
        else:
            output_mask = np.zeros((self.input_height, self.input_width), dtype=np.uint8) + 255
        
        if self.task == 'seg':
            edge = show_edge(output_mask)
            # edge_blur = np.uint8(cv2.blur(edge, (5,5)))/255.0
            return input_ori, input, edge, output_mask
            
    def __len__(self):
        return len(self.imgIds)