###  CNN 卷积神经网络  ###
# 主要：卷积层，池化层，全链接层
# 网络层构成如下：
# image
# convolution
# max pooling
# convolution
# max pooling
# fully connected
# fully connected
# classifier

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1}) # keep_prob = 1, 神经元被选中的概率为1
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1)) # tf.equal(A,B) 对比两个矩阵的元素，并返回矩阵，元素相等，返回元素true
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 将correct_prediction转化为float32格式
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result

def weight_variable(shape):
    # tf.truncated_normal(shape, mean, stddev) shape表示生成张量的维度，mean是均值，stddev是标准差
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial) 

def conv2d(x, W):
    # stide [1,x_movement,y_movement,1]
    # must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') # 中间两个为步长# tensorflow 自带的conv2d函数

def max_pool_2x2(x):
    # stride 2*2
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) #28*28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1]) # shape中-1含义是不用自己指定一维的大小，函数自动计算

# print(x_image.shap) # [n_samples, 28, 28, 1]

#################################################
# conv1 layer  高度由1变为32
W_conv1 = weight_variable([5,5,1,32]) # 卷积核patch 5x5, 拉长 输入1：in size 1, 高32：out size 32
b_conv1 = bias_variable([32]) # 输出 
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) # 非线性化处理 output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)       # output size 14x14x32  池化再缩小

# conv2 layer  高度再增加一倍
W_conv2 = weight_variable([5,5,32,64]) # 卷积核patch 5x5, 拉长 输入1：in size 1, 高32：out size 32
b_conv2 = bias_variable([64]) # 输出 
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) # 非线性化处理 output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)       # output size 7x7x64 

##########################################################
## fully connected layer
# func1 layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # 输入与权重相乘,输出维度1024行矩阵
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob) # dropout 处理，keep_prob 前方定义

# func2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

######################################################
# the error between prediction and real data
# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# init
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000],mnist.test.labels[:1000]
        ))



