#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:56:43 2018
tool to make datasets

@author: chenhp
"""
import scipy.io as si
import os 

import numpy as np
import tensorflow as tf
def make_dataset():
    test_image_name=os.listdir('H:/NR_LIVE_IQA/datasets/data1/test')
    train_image_name=os.listdir('H:/NR_LIVE_IQA/datasets/data1/train')
    for name in test_image_name:
        data=si.loadmat('H:/NR_LIVE_IQA/datasets/data1/test/'+name)
        dis_image=np.asarray(data['imdis'],np.uint8)
        label=data['label'][0,0]
        si.savemat('F:/new_IQA/MyProject1/LiveDataSets/test/'+name,{'imdis':dis_image,'label':label})
    
    for name in train_image_name:
        data=si.loadmat('H:/NR_LIVE_IQA/datasets/data1/train/'+name)
        dis_image=np.asarray(data['imdis'],np.uint8)
        label=data['label'][0,0]
        si.savemat('F:/new_IQA/MyProject1/LiveDataSets/train/'+name,{'imdis':dis_image,'label':label})
def count_Params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Total parameters are: %d'%(total_parameters))

#    return total_parameters    

 

    
    
