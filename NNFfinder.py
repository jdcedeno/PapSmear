import tensorflow as tf
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from matplotlib.pyplot import figure, imshow, show
from os import listdir


def batch(x_data, y_data, batch_size=100):
    num_samples = np.shape(x_data)[0]
    perm0 = np.arange(num_samples)
    np.random.shuffle(perm0)
    return x_data[perm0[0:batch_size]], y_data[perm0[0:batch_size]]


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def deepnn(x):
    # First Conv Layer
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    print("\n\n\n\n\n", np.shape(h_conv1), "\n\n\n\n\n")
    # First pool layer
    # Input size: 64x64x32
    # Output size: 32x32x32
    h_pool1 = max_pool_2x2(h_conv1)
    print("\n\n\n\n\n", np.shape(h_pool1), "\n\n\n\n\n")
    # Second Conv Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    print("\n\n\n\n\n", np.shape(h_conv2), "\n\n\n\n\n")
    # Second pool layer
    # Input size: 16x16x64
    # Output size: 8x8x64
    h_pool2 = max_pool_2x2(h_conv2)
    print("\n\n\n\n\n", np.shape(h_pool2), "\n\n\n\n\n")
    # Fully connected layer 1
    W_fc1 = weight_variable([16 * 16 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Fully connected layer 2
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv
# ==================================================== LOAD DATA ===================================================== #
# 56 positive, 83 negative
x_pos = []
x_neg = []

new_size = np.ones(shape=(256, 256, 3), dtype=np.uint8)

for sample in listdir("C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\Chambers Test Images\\Neuralnet\\"
                      "TrainingSamplesPositive"):
    img = imread("C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\Chambers Test Images\\Neuralnet\\"
                 "TrainingSamplesPositive\\" + sample)
    new_img = np.copy(new_size)
    new_img[:np.shape(img)[0], :np.shape(img)[1], :] = img
    x_pos.append(new_img)

for sample in listdir("C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\Chambers Test Images\\Neuralnet\\"
                      "TrainingSamplesNegative"):
    img = imread("C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\Chambers Test Images\\Neuralnet\\"
                 "TrainingSamplesNegative\\" + sample)
    new_img = np.copy(new_size)
    new_img[:np.shape(img)[0], :np.shape(img)[1], :] = img
    x_neg.append(new_img)

x_pos = np.array(x_pos)
x_neg = np.array(x_neg)

x_pos_train = np.copy(x_pos)
x_neg_train = np.copy(x_neg)

# x_pos_train = x_pos[0:44]                  # 78% of the samples are used in training (around 223 pos and 260 neg)
# x_neg_train = x_neg[0:65]
#
# x_pos_test = x_pos[44:52]                 # 14% of the samples are used for testing (41 pos and 47 neg)
# x_neg_test = x_neg[65:77]
#
# x_pos_val = x_pos[52:]                     # The rest is used for the validation set
# x_neg_val = x_neg[77:]

x_pos_train = np.repeat(x_pos_train, 100, axis=0)
x_neg_train = np.repeat(x_neg_train, 100, axis=0)

y_pos_train = np.concatenate((np.ones((np.shape(x_pos_train)[0], 1)), np.zeros((np.shape(x_pos_train)[0], 1))),
                             axis=1)
y_neg_train = np.concatenate((np.zeros((np.shape(x_neg_train)[0], 1)), np.ones((np.shape(x_neg_train)[0], 1))),
                             axis=1)

x_data = np.concatenate((x_pos_train, x_neg_train), axis=0)
y_data = np.concatenate((y_pos_train, y_neg_train), axis=0)
# ==================================================================================================================== #
# =================================== IMPLEMENT CONVOLUTIONAL NEURAL NETWORK ========================================= #
x = tf.placeholder(tf.float32, [None, 256, 256, 3])

y_ = tf.placeholder(tf.float32, [None, 2])

y_conv = deepnn(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                        logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)

accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        x_data_batch, y_data_batch = batch(x_data, y_data, batch_size=100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: x_data_batch, y_: y_data_batch})
            print('step {}, training accuracy {}'.format(i, train_accuracy))
        train_step.run(feed_dict={x: x_data_batch, y_: y_data_batch})
    print('test accuracy {}'.format(accuracy.eval(feed_dict={
        x: x_data, y_: y_data})))


