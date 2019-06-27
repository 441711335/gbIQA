import tensorflow as tf
import numpy as np
import scipy.stats as ss
import __init__
from Vgg import vgg_feature_extrate 
#from RsNet import rsnet
from ReadData import read_data
from Tools import count_Params
batch_size=32
lr=0.0001
train_path='./LiveDataSets/train/'
test_path='./LiveDataSets/test/'
size=156
data_ref,data_dis,train_label=read_data(train_path,batch_size,size)
test_ref,test_dis,test_label=read_data(test_path,batch_size,size)

x_ref=tf.placeholder(tf.float32,shape=[None,size,size,3])
x_dis=tf.placeholder(tf.float32,shape=[None,size,size,3])
y=tf.placeholder(tf.float32,shape=[None,])

def minibatches(inputs=None,inputs2=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt],inputs2[excerpt], targets[excerpt]
        
#r=rsnet(x_ref,batch_size,size)

with tf.variable_scope('feature_extrcrate') as scope:
    meanf,varf,feature_f=vgg_feature_extrate(x_ref)
    scope.reuse_variables()
    meand,vard,feature_d=vgg_feature_extrate(x_dis)
#f1=meanf-meand
f2=varf-vard
f3=feature_f-feature_d
features=tf.concat([f2,f3],axis=3)
reshape2=tf.reshape(features,[-1,512*12*12])
fc1 = tf.layers.dense(inputs=reshape2, 
                      units=512, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
fc2=tf.layers.dense(inputs=fc1, 
                      units=1, 
                      activation=None,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
output=tf.reduce_sum(tf.multiply(fc2,fc2))/tf.reduce_sum(fc2)
#output1=tf.reduce_mean(fc2)        
mean_score=tf.reduce_mean(y)
loss=tf.reduce_mean(tf.abs(output-mean_score))


f=open('./logs/logwa.txt','a')
saver=tf.train.Saver(max_to_keep=1)

b=0
c=0
train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
n_epoch=1000
#min_loss=100

sess=tf.InteractiveSession()  

sess.run(tf.global_variables_initializer())#初始化所有变量
count_Params()
for epoch in range(n_epoch):
    results=[]
    label1=[]
    if epoch==350:
        lr=lr*0.1
    #training
    print('training steps : %d'%(epoch))
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a,x_train_b, y_train_a in minibatches(data_ref, data_dis,train_label, batch_size, shuffle=False):
        _,err=sess.run([train_op,loss], feed_dict={x_ref: x_train_a, x_dis:x_train_b,y: y_train_a})
        train_loss += err; n_batch += 1
    print("   train loss: %f" % (train_loss/ n_batch))
    f.writelines("   train loss: %f" % (train_loss/ n_batch)+'\n')
    val_loss,  n_batch = 0, 0
    for x_val_a,x_val_b, y_val_a in minibatches(test_ref, test_dis,test_label, batch_size, shuffle=False):
        err,result,score=sess.run([loss,output,mean_score], feed_dict={x_ref: x_val_a, x_dis:x_val_b,y: y_val_a})
        results.append(result)
        label1.append(score)
        val_loss += err; n_batch += 1
    print("   test loss: %f" % (val_loss/ n_batch))
    f.writelines("   test loss: %f" % (val_loss/ n_batch)+'\n')
    a=ss.spearmanr(label1,results)
    a1=ss.pearsonr(label1,results)
    if a1[0]>c:
        c=a1[0]
    f.writelines('Best PLCC is :%f'%c+'\n')
    if a[0]>b:
        b=a[0]
        saver.save(sess,'./models/Wafrmodel/model.ckpt')
    f.writelines('Best SROCC is :%f'%b+'\n')
    print('Best SROCC is :%f'%b)
    print('Best PLCC is :%f'%c)
sess.close()
f.close()
#import scipy as si
#data=si.io.loadmat('./LiveDataSets/test/fastfadingimg1.mat')
#l=data['label'][0,0]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    