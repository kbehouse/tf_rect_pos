import tensorflow as tf


def weight_variable(shape):
    print('weight_variable shape = ' + str(shape))
    inital = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

def Conv2D(x , kernel_size = 3, out_channel = 32, in_channel = None):
    if in_channel is None:
        assert len(x.shape) == 4, 'Conv2D() say the len of input shape is not 4'
        in_channel = int(x.shape[3])

    # print('Conv2D in_channel = %d' % in_channel)
    # w and b
    w = weight_variable([kernel_size, kernel_size, in_channel, out_channel]) 
    b = bias_variable([out_channel])
    
    #Combine
    return tf.nn.relu(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')+ b) #output size 28x28x32


def MaxPool2D(x, pool_size = 2):
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                        strides=[1, pool_size, pool_size, 1], padding='SAME')

def Flaten(x):
    assert len(x.shape) == 4, 'flat() say the len of input shape is not 4'
    num = int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3])
    # print('flat num = %d' % num)
    return tf.reshape(x, [-1, num]) 

def FC(x, fc_size = 1024):
    assert len(x.shape) == 2, 'FC() say the len of input shape is not 2'
    num = int(x.shape[1]) 

    w = weight_variable([num, fc_size])
    b = bias_variable([fc_size])

    return tf.nn.relu(tf.matmul(x, w) + b)


def Output(x, out_size = 10):
    assert len(x.shape) == 2, 'Output() say the len of input shape is not 2'
    num = int(x.shape[1]) #* int(x.shape[2]) * int(x.shape[3])
    
    w = weight_variable([num, out_size])
    b = bias_variable([out_size])
    return tf.nn.softmax(tf.matmul(x, w) + b)