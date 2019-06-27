import tensorflow as tf

def vgg_feature_extrate(x):
    conv1_weight=tf.get_variable(name='Conv1_weightS',shape=[3,3,3,32],initializer=tf.contrib.layers.xavier_initializer())
    conv1_bias=tf.get_variable(name='Conv1_bias',shape=[32],initializer=tf.constant_initializer(value=0.))
    conv1=tf.nn.conv2d(x,conv1_weight,strides=[1,1,1,1],padding='VALID')
    conv1=tf.nn.relu(conv1+conv1_bias)
#    print(conv1)
    
    conv2_weight=tf.get_variable(name='Conv2_weightS',shape=[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer())
    conv2_bias=tf.get_variable(name='Conv2_bias',shape=[32],initializer=tf.constant_initializer(value=0.))
    conv2=tf.nn.conv2d(conv1,conv2_weight,strides=[1,1,1,1],padding='VALID')
    conv2=tf.nn.relu(conv2+conv2_bias)
    
    pool1=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    conv3_weight=tf.get_variable(name='Conv3_weightS',shape=[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer())
    conv3_bias=tf.get_variable(name='Conv3_bias',shape=[64],initializer=tf.constant_initializer(value=0.))
    conv3=tf.nn.conv2d(pool1,conv3_weight,strides=[1,1,1,1],padding='VALID')
    conv3=tf.nn.relu(conv3+conv3_bias)
    
    conv4_weight=tf.get_variable(name='Conv4_weightS',shape=[3,3,64,64],initializer=tf.contrib.layers.xavier_initializer())
    conv4_bias=tf.get_variable(name='Conv4_bias',shape=[64],initializer=tf.constant_initializer(value=0.))
    conv4=tf.nn.conv2d(conv3,conv4_weight,strides=[1,1,1,1],padding='VALID')
    conv4=tf.nn.relu(conv4+conv4_bias)
    
    pool2=tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    conv5_weight=tf.get_variable(name='Conv5_weightS',shape=[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer())
    conv5_bias=tf.get_variable(name='Conv5_bias',shape=[128],initializer=tf.constant_initializer(value=0.))
    conv5=tf.nn.conv2d(pool2,conv5_weight,strides=[1,1,1,1],padding='VALID')
    conv5=tf.nn.relu(conv5+conv5_bias)
    
    conv6_weight=tf.get_variable(name='Conv6_weightS',shape=[3,3,128,128],initializer=tf.contrib.layers.xavier_initializer())
    conv6_bias=tf.get_variable(name='conv6_bias',shape=[128],initializer=tf.constant_initializer(value=0.))
    conv6=tf.nn.conv2d(conv5,conv6_weight,strides=[1,1,1,1],padding='VALID')
    conv6=tf.nn.relu(conv6+conv6_bias)
    
    pool3=tf.nn.max_pool(conv6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    conv7_weight=tf.get_variable(name='Conv7_weightS',shape=[3,3,128,256],initializer=tf.contrib.layers.xavier_initializer())
    conv7_bias=tf.get_variable(name='Conv7_bias',shape=[256],initializer=tf.constant_initializer(value=0.))
    conv7=tf.nn.conv2d(pool3,conv7_weight,strides=[1,1,1,1],padding='VALID')
    conv7=tf.nn.relu(conv7+conv7_bias)
    
    conv8_weight=tf.get_variable(name='Conv8_weightS',shape=[3,3,256,256],initializer=tf.contrib.layers.xavier_initializer())
    conv8_bias=tf.get_variable(name='Conv8_bias',shape=[256],initializer=tf.constant_initializer(value=0.))
    conv8=tf.nn.conv2d(conv7,conv8_weight,strides=[1,1,1,1],padding='VALID')
    conv8=tf.nn.relu(conv8+conv8_bias)
    
    mean=tf.multiply(tf.reduce_mean(conv8,reduction_indices=[1,2],keepdims=True),tf.ones_like(conv8))
    var=tf.square(conv8-mean)

    return mean,var,conv8