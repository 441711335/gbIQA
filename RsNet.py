import tensorflow as tf

def rsnet(x,batch_size,size):
    conv1_weight=tf.get_variable(name='conv1_weight',shape=[5,5,3,32],initializer=tf.contrib.layers.xavier_initializer())
    conv1_bias=tf.get_variable(name='conv1_bias',shape=[32],initializer=tf.constant_initializer(value=0.))
    conv1=tf.nn.conv2d(x,conv1_weight,strides=[1,1,1,1],padding='VALID')
    conv1=tf.nn.relu(conv1+conv1_bias)
#    print(conv1)
    pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    conv2_weight=tf.get_variable(name='conv2_weight',shape=[5,5,32,64],initializer=tf.contrib.layers.xavier_initializer())
    conv2_bias=tf.get_variable(name='conv2_bias',shape=[64],initializer=tf.constant_initializer(value=0.))
    conv2=tf.nn.conv2d(pool1,conv2_weight,strides=[1,1,1,1],padding='VALID')
    conv2=tf.nn.relu(conv2+conv2_bias)
    
    pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    conv3_weight=tf.get_variable(name='conv3_weight',shape=[5,5,64,128],initializer=tf.contrib.layers.xavier_initializer())
    conv3_bias=tf.get_variable(name='conv3_bias',shape=[128],initializer=tf.constant_initializer(value=0.))
    conv3=tf.nn.conv2d(pool2,conv3_weight,strides=[1,1,1,1],padding='VALID')
    conv3=tf.nn.relu(conv3+conv3_bias)
    
    pool3=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    
    deconv1_weight=tf.get_variable(name='deconv1_weight',shape=[5,5,64,128],initializer=tf.contrib.layers.xavier_initializer())
    deconv1=tf.nn.conv2d_transpose(pool3,filter=deconv1_weight,strides=[1,2,2,1],output_shape=[batch_size,size//4-3,size//4-3,64],padding='VALID')
    
    deconv2_weight=tf.get_variable(name='deconv2_weight',shape=[5,5,32,64],initializer=tf.contrib.layers.xavier_initializer())
    deconv2=tf.nn.conv2d_transpose(deconv1,filter=deconv2_weight,strides=[1,2,2,1],output_shape=[batch_size,size//2-2,size//2-2,32],padding='VALID')
    
    deconv3_weight=tf.get_variable(name='deconv3_weight',shape=[5,5,3,32],initializer=tf.contrib.layers.xavier_initializer())
    deconv3=tf.nn.conv2d_transpose(deconv2,filter=deconv3_weight,strides=[1,2,2,1],output_shape=[batch_size,size,size,3],padding='VALID')
    return deconv3