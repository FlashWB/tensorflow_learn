# http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_pros.html
# https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf16_classification/full_code.py
# 进阶版 https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/404_AutoEncoder.py

from __future__ import print_function
import tensorflow as tf
# 需要翻墙
import tensorflow.examples.tutorials.mnist.input_data as input_data
# number 1- 10
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer (inputs, in_size, out_size, activation_function=None,):
    #add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 加速
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1)) # tf.argmax()按行和列求矩阵最大值，0代表列，1代表行
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # tf.cost(x, dytpe, name = None) 将 数据 x 的格式转换为dtype
    result = sess.run(accuracy, feed_dict={xs:v_xs,ys:v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784]) # 28 * 28 =784 个像素点
ys = tf.placeholder(tf.float32,[None,10])  # output 1-10

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)  # 图像处理使用softmax,以前做回归可以不用

# the emeanrror between prediction and real data
# https://www.zhihu.com/question/41252833 交叉熵讲解
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1])) # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())

# start
for i in range(2000):
    #每次学习 100 个
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels
        ))




















