
# Get 99% accuracy from tensorflow tutorial #

import tensorflow as tf
from DNN import *
import pickle
from rect_env import RectEnv
# import cv2
from scipy.misc import imread
import numpy as np 
def euclidean(x,y):
    """
      x:      [0,0],[1,1],[2,2]
      y:      [3,4 ],[7,9],[5,5]
      output: [  5.         10.          4.2426405]
    """
    return tf.sqrt(tf.reduce_sum(tf.square(x-y),1))

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    x = x.astype('float32')
    x /= 255.0
    return x


tf.reset_default_graph()

# define placeholder for inputs to network
x_image = tf.placeholder(tf.float32, [None, 240, 240, 3]) 

ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
#-----------------Model-----------------#
conv1 = Conv2D(x_image, 4, 32)
pool1 = MaxPool2D(conv1) #output size 14x14x32
conv2 = Conv2D(pool1, 4, 64)
pool2 = MaxPool2D(conv2, 2) #output size 14x14x32
flat = Flaten (pool2)
fc1 = FC(flat)
fc1_drop = tf.nn.dropout(fc1, tf.to_float(keep_prob))
prediction = FC(flat,2)


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

env = RectEnv()
env.recreate_dir('valid_pic')

save_model_path = 'checkpoints/'
meta_path = save_model_path + "model.ckpt.meta"




with tf.Session() as sess:
    # Load model
    # loader = tf.train.import_meta_graph(meta_path)
    # loader.restore(sess, save_model_path+'model.ckpt')

    saver.restore(sess,  save_model_path+'model.ckpt')
    print("Model restored.")
    

    # for i in range(100000):
    while 1:
        env.random_rect_pos()
        env.render()
        
        f_name = env.get_last_filename()
        # print('f_name = ', f_name)
        # im = cv2.imread(f_name)
        im = imread(f_name, mode='RGB')
        pic_normal = normalize(im)
        im_newaxis = pic_normal [np.newaxis, :]
        # print('im_newaxis.shape=' + str(im_newaxis.shape) )
        pred = sess.run(prediction, feed_dict={x_image: im_newaxis, keep_prob:1.0})

        
        env.set_predict_rect_pos(pred)
        env.render()

        dis = np.sqrt(np.sum((env.target_rect_pos-pred)**2))
        print('target={}, pred = {}, dis = {}'.format(env.target_rect_pos, pred, dis) )
        

# loaded_graph = tf.Graph()

# with tf.Session(graph=loaded_graph) as sess:
#     # Load model
#     loader = tf.train.import_meta_graph(meta_path)
#     loader.restore(sess, save_model_path+'model.ckpt')

#     # saver.restore(sess,  save_model_path+'model.ckpt')
#     print("Model restored.")
    

#     for i in range(100000):
#         env.random_rect_pos()
#         env.render()
        
#         f_name = env.get_last_filename()
#         print('f_name = ', f_name)
#         im = cv2.imread(f_name)
#         pic_normal = normalize(im)
#         im_newaxis = im [np.newaxis, :]
#         print('im_newaxis.shape=' + str(im_newaxis.shape) )
#         pred = sess.run(prediction, feed_dict={x_image: im_newaxis, keep_prob:1.0})

#         print('pred = ',pred)
#         env.set_predict_rect_pos(pred)
#         env.render()
