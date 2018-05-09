import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import sys

def load_training_data(file_path):
    data = pd.read_csv(file_path)
    data = data[data.Usage == "Training"]
    pixels = data.pixels.str.split(" ").tolist()
    pixels = pd.DataFrame(pixels, dtype=int)
    return pixels.values.astype(np.float), data["emotion"].values.ravel()

def pre_processing(pixels, labels):
    #standarize pixels
    pixels = pixels - pixels.mean(axis=1).reshape(-1, 1)
    pixels = np.multiply(pixels, 100.0/255.0)
    each_pixel_mean = pixels.mean(axis=0)
    each_pixel_std = np.std(pixels, axis=0)
    pixels = np.divide(np.subtract(pixels, each_pixel_mean), each_pixel_std)
    #change labels numbers to vectors
    label_vectors = np.zeros((labels.shape[0], labels_count))
    for i in range(labels.shape[0]):
        label_vectors[i][labels[i]] = 1
    return pixels, label_vectors.astype(np.uint8)

def init_weight(shape, bLocal):
    stddev = 0.04 if bLocal else 1e-4
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def init_bias(shape, bLocal):
    init = 0.0 if bLocal else 0.1
    return tf.Variable(tf.constant(init, shape=shape))

def new_batch():
    train_index = 0
    def get_batch(total_x, total_y, batch_size):
        nonlocal train_index
        num = total_x.shape[0]
        start = train_index
        train_index += batch_size
        # shuffle training data when all data has been used
        if train_index > num:
            start = 0
            train_index = batch_size
            re_order = np.random.shuffle(np.arange(num))
            total_x = total_x[re_order]
            total_y = total_y[re_order]
        end = train_index
        return total_x[start:end], total_y[start:end]
    return get_batch

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="the input file training data path")
    parser.add_argument("-output", help="the output SavedModel path") 
    args = parser.parse_args()
    data_file = args.input
    if (not data_file):
        print("no input data")
        sys.exit()
    model_path = args.output
    if (not data_file):
        print("no output path")
        sys.exit()

    pixels, labels = load_training_data(data_file)
    pixels_amount = pixels.shape[1]
    labels_count = np.unique(labels).shape[0]
    width = height = np.ceil(np.sqrt(pixels_amount)).astype(np.uint8)
    train_pixels, train_labels = pre_processing(pixels, labels)

    #Build CNN Model
    # input:x,  output: y_
    x = tf.placeholder('float', shape=[None, pixels_amount])#(28709, 2304)
    y_ = tf.placeholder('float', shape=[None, labels_count])#(28709, 7)
    # first convolutional layer 64
    w_conv1 = init_weight([5, 5, 1, 64], False)
    b_conv1 = init_bias([64], False)
    image = tf.reshape(x, [-1, width, height, 1])#(28709,48,48,1)
    h_conv1 = tf.nn.relu(b_conv1+tf.nn.conv2d(image, w_conv1, strides=[1, 1, 1, 1], padding="SAME"))#(28709,48,48,64)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')#(28709,24,24,64)
    h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)#(28709,24,24,64)
    # second convolutional layer
    w_conv2 = init_weight([5, 5, 64, 128], False)
    b_conv2 = init_bias([128], False)
    h_conv2 = tf.nn.relu(b_conv2+tf.nn.conv2d(h_norm1, w_conv2, strides=[1, 1, 1, 1], padding="SAME"))#(28709,24,24,128)
    h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)#(28709,24,24,128)
    h_pool2 = tf.nn.max_pool(h_norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')#(28709, 12, 12, 128)
    # densely connected layer local 3
    w_fc1 = init_weight([12*12*128, 3072], True)
    b_fc1 = init_bias([3072], True)
    h_fc1 = tf.nn.relu(b_fc1+tf.matmul(tf.reshape(h_pool2, [-1, 12*12*128]), w_fc1))#(28709, 3702)
    # densely connected layer local 4
    w_fc2 = init_weight([3072, 1536], True)
    b_fc2 = init_bias([1536], True)
    h_fc2 = tf.nn.relu(b_fc2+tf.matmul(tf.reshape(h_fc1, [-1, 3072]), w_fc2))#(28709, 1536)
    # read output
    w_fc3 = init_weight([1536, labels_count], False)
    b_fc3 = init_bias([labels_count], False)
    y = tf.nn.softmax(tf.matmul(h_fc2, w_fc3) + b_fc3)  # (28709, 7)
    # prediction function
    predict = tf.argmax(y, 1)

    LEARNING_RATE = 1e-4
    # cost function
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    # optimization function
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    # evaluation
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), 'float'))

    # start TensorFlow session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    MAX_ITERATIONS = 3000
    BATCH_SIZE = 50
    iter_display = 1
    get_new_batch = new_batch()
    for i in range(MAX_ITERATIONS):
        batch_x, batch_y = get_new_batch(train_pixels, train_labels, BATCH_SIZE)
        # check performance in the process
        if (i % iter_display)==0 or i==(MAX_ITERATIONS-1):
            if (i == 10 or i == 100) and iter_display < 100:
                iter_display *= 10
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
            print('step %d: accuracy=%.2f' % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

    tf.saved_model.simple_save(sess, model_path, inputs={"x": x, "y_": y_}, outputs={"y": y})
