#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:56:43 2018
files to read data
@author: chenhaopeng
"""

import scipy.io as si
import os
import numpy as np
import cv2

def random_crop(image_ref,image_dis,num_output,size):
    h,w=image_ref.shape[:2]
    random_h=np.random.randint(h-size,size=num_output)
    random_w=np.random.randint(w-size,size=num_output) 
    patches_dis=[]
    patches_ref=[]
    for i in range(num_output):
        patch_dis=image_dis[random_h[i]:random_h[i]+size,random_w[i]:random_w[i]+size]
        patch_ref=image_ref[random_h[i]:random_h[i]+size,random_w[i]:random_w[i]+size]
        patches_ref.append(patch_ref)
        patches_dis.append(patch_dis)
    return patches_ref,patches_dis

def multif(image):
    for i in range(20):
        image=cv2.GaussianBlur(image,(7,7),1.6)
    return image

def read_data(path,bach_size,size):
    images_ref=[]
    images_dis=[]
    labels=[]
    filename=os.listdir(path)
    for fn in filename:
        data=si.loadmat(path+fn)
        print('read data :%s'%(path+fn))
        imagedis=data['imdis']
        imageref=data['imref']
        label=data['label'][0,0]
        if label!=0.:
            patches_ref,patches_dis=random_crop(imageref,imagedis,num_output=32,size=size)
            for p1 in patches_ref:
                images_ref.append(p1)
                labels.append(label)
            for p2 in patches_dis:
                images_dis.append(p2)
    return np.asarray(images_ref,np.uint8),np.asarray(images_dis,np.uint8),np.asarray(labels,np.float)

#image=cv2.imread('I:/IQADatabases/LIVE/databaserelease2/refimgs/plane.bmp')
#im1=multif(image)
#cv2.imshow('test',im1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# 