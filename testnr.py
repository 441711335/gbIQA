import tensorflow as tf
import scipy.io as scio
import scipy.stats as st
import os
import numpy as np
import __init__
#import cv2
from ReadData import random_crop
saver=tf.train.import_meta_graph('./models/nrmodel/model.ckpt.meta')
sess=tf.InteractiveSession()
saver.restore(sess,'./models/nrmodel/model.ckpt')
x_ref=tf.get_default_graph().get_tensor_by_name('Placeholder:0')
x_dis=tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
logits=tf.get_default_graph().get_tensor_by_name('Mean:0')
#g=tf.get_default_graph().get_tensor_by_name('conv2d_transpose_1:0')
datanames1=os.listdir('./LiveDataSets/test/')
datanames2=os.listdir('./LiveDataSets/train/')
score=[]
p_score=[]
#p_score1=[]
#img1=[]
#def multif(image):
#    for i in range(20):
#        image=cv2.GaussianBlur(image,(7,7),1.6)
#    return image
for im in datanames1:
    data=scio.loadmat('./LiveDataSets/test/'+im)
    print('load mat of :'+'./LiveDataSets/test/'+im)
    imdis=data['imdis']
    imref=data['imdis']
    
    label=str(data['label'][0,0])
    score.append(data['label'][0,0])
    patches_ref,patches_dis=random_crop(imref,imdis,num_output=32,size=156)
    im1=np.asarray(patches_ref,np.uint8) 
    im2=np.asarray(patches_dis,np.uint8)
    predict=sess.run(logits,feed_dict={x_dis:im2,x_ref:im1})
    p_score.append(predict)
for im in datanames2:
    data=scio.loadmat('./LiveDataSets/train/'+im)
    print('load mat of :'+'./LiveDataSets/train/'+im)
    imdis=data['imdis']
    imref=data['imdis']
    
    label=str(data['label'][0,0])
    score.append(data['label'][0,0])
    patches_ref,patches_dis=random_crop(imref,imdis,num_output=32,size=156)
    im1=np.asarray(patches_ref,np.uint8) 
    im2=np.asarray(patches_dis,np.uint8)
    predict=sess.run(logits,feed_dict={x_dis:im2,x_ref:im1})
    p_score.append(predict)
SROCC=st.spearmanr(score,p_score)[0]  
PLCC=st.pearsonr(score,p_score)[0]  
print('SROCC is: %s,and PLCC is: %s'%(SROCC,PLCC))
sess.close()