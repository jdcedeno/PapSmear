import numpy as np
import tensorflow as tf
import pickle
from skimage.io import imread
from matplotlib.pyplot import figure, imshow, show
from matplotlib.pyplot import hlines, vlines
from os import listdir, chdir
from os.path import abspath
from tkinter.filedialog import askdirectory


def next_batch():
    """
    This function looks for the pictures in the specified folders, and randomly selects 50 samples along with their
    labels.
    :return: result: a tuple containing (samples_neg, labels_neg). samples_neg is a (50, 1536, 2048, 3) tensor
    containing 50 samples. labels_neg is a (50, 96, 128, 5) tensor containing the labels for each given image.
    The format for samples_neg: (batch, height, width, channels)
    The format for labels_neg is: (batch, width, height, bounding_box)
    bounding_box contains information about the bounding box: [class, bx, by, bh, bw]
    class: the only class, can be 1 if there is an object in the box or 0 if there isn't an object in that box
    """
    samples_neg = []
    labels_neg = []
    shuffled_folder_index = tuple(np.random.randint(2, 10, 10))
    for counts in shuffled_folder_index:
        directory = 'C:/Users/Owner/PycharmProjects/ProteanLabs/PapSmear/neg {}/output'.format(counts)
        label_path = 'C:/Users/Owner/PycharmProjects/ProteanLabs/PapSmear/neg {}/Y_neg_{}.npy'.format(counts, counts)
        label = np.load(label_path)
        # label_all = np.repeat(label, 5, 1)
        label_all = label
        labels_neg.append(label_all)
        shuffled_sample_index = tuple(np.random.randint(0, 100, 1))
        path_list = listdir(directory)
        for count2 in shuffled_sample_index:
            path = path_list[count2]
            full_path = directory + '/' + path
            image = imread(full_path)
            image = image / 255.
            samples_neg.append(image)
    labels_neg = np.array(labels_neg).reshape(-1, 96, 128, 5)
    result = (samples_neg, labels_neg)
    return result


# x, y = next_batch()
# print('x shape is: ', np.shape(x))
# print('y shape is: ', np.shape(y))
# figure('255')
# imshow(x[0])
# figure('1')
# norm_img = x[0]/255.
# imshow(norm_img)
# show()
pass

# ============================================= training data arrays ================================================= #
# samples = []
# print(np.shape(samples))
# directory = askdirectory()
# for path in listdir(directory):
#     full_path = directory + '/' + path
#     image = imread(full_path)
#     samples.append(image)
# print(np.shape(samples))
#
# save_dir = 'C:/Users/Owner/PycharmProjects/ProteanLabs/PapSmear/training_data_neg/samples/' + 'neg_9_train.npy'
# np.save(save_dir, samples)
# ==================================================================================================================== #

pass

# ===================================================== Model ======================================================== #
X = tf.placeholder(tf.float32, [None, 1536, 2048, 3])
Y = tf.placeholder(tf.float32, [None, 96, 128, 5])

W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
W3 = tf.get_variable("W3", [4, 4, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
W4 = tf.get_variable("W4", [2, 2, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
W5 = tf.get_variable("W5", [1, 1, 64, 5], initializer=tf.contrib.layers.xavier_initializer(seed=0))

Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
A1 = tf.nn.relu(Z1)
P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
A2 = tf.nn.relu(Z2)
P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
A3 = tf.nn.relu(Z3)
P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')
A4 = tf.nn.relu(Z4)
P4 = tf.nn.max_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Z5 = tf.nn.conv2d(P4, W5, strides=[1, 1, 1, 1], padding='SAME')
A5 = tf.nn.relu(Z5)
A6 = tf.nn.softmax(A5)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=A6))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for count in range(1000):
    print('count: ', count)
    x_feed, y_feed = next_batch()
    a = sess.run(train_step, {X: x_feed, Y: y_feed})
    print('count again: ', count)
print("exit count")
save_path = saver.save(sess, "C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\PapSmear\\Models\\model_1.ckpt")
val_sample = np.array(imread("C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\PapSmear\\neg 2\\neg 2.jpg"))
val_sample = val_sample/255.
val_sample.reshape([1, 1536, 2048, 3])
print('val_sample.reshape: ', np.shape(val_sample))
val_label = np.load("C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\PapSmear\\neg 2\\Y_neg_2.npy")
validation = sess.run(A6, {X: val_sample, Y: val_label})
print("validation result is: ", validation)

pass

# =============================================== create labels ====================================================== #

# image = imread("C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\PapSmear\\neg 9\\neg 9.jpg")
# fig_neg_9 = figure("neg 9")
# fig_neg_9_ax = fig_neg_9.gca()
# fig_neg_9_ax.imshow(image)
#
# horizontal = []
# vertical = []
# for count in range(1, 96):
#     num_1536 = count * 16
#     horizontal.append(num_1536)
# for count in range(1, 128):
#     num_2048 = count * 16
#     vertical.append(num_2048)
#
# fig_neg_9_ax.hlines(horizontal, 0, 2048)
# fig_neg_9_ax.vlines(vertical, 0, 1536)
# fig_neg_9_ax.set_xlim(0, 2048)
# fig_neg_9_ax.set_ylim(1536, 0)
# fig_neg_9.savefig("C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\PapSmear\\neg 9\\fig_neg_9")
# #
# # show()
#
# pickle_in = open("C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\PapSmear\\neg 9\\neg 9_box_labels.pickle", "rb")
# neg_9_coordinates = pickle.load(pickle_in)
#
# Y_neg_9 = np.zeros((1, 96, 128, 5))
# Y_neg_9_image = np.ones((96, 128))
# for coord in neg_9_coordinates:
#     center_x = coord["center_x"]
#     center_y = coord["center_y"]
#     height = coord["height"] / 16
#     width = coord["width"] / 16
#     grid_x = np.floor(center_x / 16)
#     grid_y = np.floor(center_y / 16)
#     center_x_norm = (center_x / 16) - grid_x
#     center_y_norm = (center_y / 16) - grid_y
#
#     Y_neg_9[0, int(grid_y), int(grid_x), :] = [1, center_y_norm, center_x_norm, height, width]
#     Y_neg_9_image[int(grid_y), int(grid_x)] = 0
#
# np.save("C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\PapSmear\\neg 9\\Y_neg_9.npy", Y_neg_9)
#
#
# Y_neg_9_image_figure = figure("Y_image")
# imshow(Y_neg_9_image, cmap='gray')
# Y_neg_9_image_figure.savefig("C:\\Users\\Owner\\PycharmProjects\\ProteanLabs\\PapSmear\\neg 9\\Y_neg_9_image_figure")

# show()

# ==================================================================================================================== #









