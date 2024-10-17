
import cv2
import math
import random
import numpy as np
from PIL import ImageEnhance

# global parameter
set_ratio = 0.5

def aug_matrix(img_w, img_h, bbox, w, h, angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=40):
    ''' 
    first Translation, then rotate, final scale.
        [sx, 0, 0]       [cos(theta), -sin(theta), 0]       [1, 0, dx]       [x]
        [0, sy, 0] (dot) [sin(theta),  cos(theta), 0] (dot) [0, 1, dy] (dot) [y]
        [0,  0, 1]       [         0,           0, 1]       [0, 0,  1]       [1]
    '''
    ratio = 1.0*(bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(img_w*img_h)
    x_offset = (random.random()-0.5) * 2 * offset
    y_offset = (random.random()-0.5) * 2 * offset
    dx = (w-(bbox[2]+bbox[0]))/2.0 
    dy = (h-(bbox[3]+bbox[1]))/2.0
    
    matrix_trans = np.array([[1.0, 0, dx],
                             [0, 1.0, dy],
                             [0, 0,   1.0]])

    angle = random.random()*(angle_range[1]-angle_range[0])+angle_range[0]
    scale = random.random()*(scale_range[1]-scale_range[0])+scale_range[0]
    scale *= np.mean([float(w)/(bbox[2]-bbox[0]), float(h)/(bbox[3]-bbox[1])])
    alpha = scale * math.cos(angle/180.0*math.pi)
    beta = scale * math.sin(angle/180.0*math.pi)

    centerx = w/2.0 + x_offset
    centery = h/2.0 + y_offset
    H = np.array([[alpha, beta, (1-alpha)*centerx-beta*centery], 
                  [-beta, alpha, beta*centerx+(1-alpha)*centery],
                  [0,         0,                            1.0]])

    H = H.dot(matrix_trans)[0:2, :]
    return H  

# ===================== normalization for input image =====================
def padding(img_ori, mask_ori, size=224, padding_color=128):
    height = img_ori.shape[0]
    width = img_ori.shape[1]
    
    img = np.zeros((max(height, width), max(height, width), 3)) + padding_color
    mask = np.zeros((max(height, width), max(height, width)))
    
    if (height > width):
        padding = int((height-width)/2)
        img[:, padding:padding+width, :] = img_ori
        mask[:, padding:padding+width] = mask_ori
    else:
        padding = int((width-height)/2)
        img[padding:padding+height, :, :] = img_ori
        mask[padding:padding+height, :] = mask_ori
        
    img = np.uint8(img)
    mask = np.uint8(mask)
    
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_CUBIC)
    
    return np.array(img, dtype=np.float32),  np.array(mask, dtype=np.float32)

def Normalize_Img(imgOri, scale, mean, val):
    img = np.array(imgOri.copy(), np.float32)/scale
    if len(img.shape) == 4:
        for j in range(img.shape[0]):
            for i in range(len(mean)):
                img[j,:,:,i] = (img[j,:,:,i]-mean[i])*val[i]
        return img
    else:
        for i in range(len(mean)):
            img[:,:,i] = (img[:,:,i]-mean[i])*val[i]
        return img

def Anti_Normalize_Img(imgOri, scale, mean, val):
    img = np.array(imgOri.copy(), np.float32)
    if len(img.shape) == 4:
        for j in range(img.shape[0]):
            for i in range(len(mean)):
                img[j,:,:,i] = img[j,:,:,i]/val[i]+mean[i]
        return np.array(img*scale, np.uint8)
    else:
        for i in range(len(mean)):
            img[:,:,i] = img[:,:,i]/val[i]+mean[i]
        return np.array(img*scale, np.uint8)
        
def data_aug_flip(image, mask):
    if random.random()<set_ratio:
        return image, mask, False
    return image[:,::-1,:], mask[:,::-1], True

def data_aug_color(image):  
    if random.random()<set_ratio:
        return image
    random_factor = np.random.randint(4, 17) / 10. 
    color_image = ImageEnhance.Color(image).enhance(random_factor) 
    random_factor = np.random.randint(4, 17) / 10. 
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
    random_factor = np.random.randint(6, 15) / 10. 
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = np.random.randint(8, 13) / 10.
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

def data_aug_blur(image):
    if random.random()<set_ratio:
        return image
    
    select = random.random()
    if select < 0.3:
        kernalsize = random.choice([3,5])
        image = cv2.GaussianBlur(image, (kernalsize,kernalsize),0)
    elif select < 0.6:
        kernalsize = random.choice([3,5])
        image = cv2.medianBlur(image, kernalsize)
    else:
        kernalsize = random.choice([3,5])
        image = cv2.blur(image, (kernalsize,kernalsize))
    return image

def data_aug_noise(image):
    if random.random()<set_ratio:
        return image
    mu = 0
    sigma = random.random()*10.0
    image = np.array(image, dtype=np.float32)
    image += np.random.normal(mu, sigma, image.shape)
    image[image>255] = 255
    image[image<0] = 0
    return image

# ===================== generate edge for input image =====================
def show_edge(mask_ori):
    mask = mask_ori.copy()
    # find countours: img must be binary
    myImg = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    ret, binary = cv2.threshold(np.uint8(mask)*255, 127, 255, cv2.THRESH_BINARY)
    countours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL
    # img, countours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL
    '''
    cv2.drawContours(myImg, countours, -1, 1, 10)
    diff = mask + myImg
    diff[diff < 2] = 0
    diff[diff == 2] = 1
    return diff   
    '''
    cv2.drawContours(myImg, countours, -1, 1, 4)
    return myImg