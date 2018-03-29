
# Get 99% accuracy from tensorflow tutorial #

import tensorflow as tf
from DNN import *
import pickle

def euclidean(x,y):
    """
      x:      [0,0],[1,1],[2,2]
      y:      [3,4 ],[7,9],[5,5]
      output: [  5.         10.          4.2426405]
    """
    return tf.sqrt(tf.reduce_sum(tf.square(x-y),1))

#---------------Load Data------------------#
def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    print('in batch_features_labels')
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'batch_%02d.p' % batch_id
    features, labels = pickle.load(open(filename, mode='rb'))
    print('in load_preprocess_training_batch load batch_id=', batch_id)

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


# define placeholder for inputs to network
x_image = tf.placeholder(tf.float32, [None, 240, 240, 3]) 

ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
#-----------------Model-----------------#
conv1 = Conv2D(x_image, 4, 32)
pool1 = MaxPool2D(conv1) 
conv2 = Conv2D(pool1, 4, 64)
pool2 = MaxPool2D(conv2, 2)
flat = Flaten (pool2)
fc1 = FC(flat)
fc1_drop = tf.nn.dropout(fc1, tf.to_float(keep_prob))
prediction = FC(flat,2)

# conv1 = Conv2D(x_image, 4, 32)
# # conv2 = Conv2D(conv1, 4, 32)
# flat = Flaten (conv1)
# fc1 = FC(flat)
# fc1_drop = tf.nn.dropout(fc1, tf.to_float(keep_prob))
# prediction = FC(flat,2)



dis = euclidean(prediction, ys)
cost = tf.reduce_mean(dis)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)


# sess = tf.Session()
# sess.run(tf.initialize_all_variables())

batch_size = 64

def train_all_batch(sess, epoch, prefer_loss = 1.0, batch_size = 64):
    for batch_i in range(10):
        sum_pattern = 0
        for batch_imgs, batch_pos in load_preprocess_training_batch(batch_i, batch_size):
            sum_pattern += len(batch_imgs)

            sess.run(train_step, feed_dict={x_image: batch_imgs, ys: batch_pos, keep_prob:0.5})
            pred,loss = sess.run( (prediction,cost), feed_dict={x_image: batch_imgs, ys: batch_pos, keep_prob:1.0})
            pred = sess.run(prediction, feed_dict={x_image: batch_imgs, keep_prob:1.0})

            # print(pred[0])
            print("epoch: {:2d}, batch: {:2d}, sum_pattern: {:4d}, loss: {:4.2f}" \
                .format(epoch, batch_i, sum_pattern, loss))

            if loss < prefer_loss:
                return loss
    return loss



# save_model_path = 'save_model/'
save_model_path = 'checkpoints/'
want_loss = 1.5
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    import time
    train_start_time = time.time()
    

    for ep in range(100):
        if train_all_batch(sess,ep, prefer_loss = want_loss) < want_loss:
            break
        print("Training Time: ", time.time() - train_start_time, " seconds.")

    print("Training Time: ", time.time() - train_start_time, " seconds.")

    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path+'model.ckpt')

    # for batch_i in range(10):
    #     sum_pattern = 0
    #     for batch_imgs, batch_pos in load_preprocess_training_batch(batch_i, batch_size):
    #         sum_pattern += len(batch_imgs)

    #         sess.run(train_step, feed_dict={x_image: batch_imgs, ys: batch_pos, keep_prob:0.5})
    #         pred,loss = sess.run( (prediction,cost), feed_dict={x_image: batch_imgs, ys: batch_pos, keep_prob:1.0})
    #         # predic = sess.run(prediction, feed_dict={x_image: batch_imgs, keep_prob:1.0})
            
    #         print("batch: {:2d}, sum_pattern: {:4d}, Loss: {:4.2f}".format(batch_i, sum_pattern, loss))
