###  activation function  ###
#达到某个值，被激活activated，deactivated, 筛选
#tensorflow uses activation function before output

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#构造一个神经层函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  
    # bias是函数的截距
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#导入数据,带噪声noise的一元二次函数
x_data = np.linspace(-1,1,300, dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

#tf.placeholder，使用占位符定义神经网络的输入.None代表无论输入多少个，输入只有一个特征，所以为1
xs = tf.placeholder(tf.float32,[None,1],name='x_input')
ys = tf.placeholder(tf.float32,[None,1],name='y_input')

# 构造  输入层1个神经元  隐藏层 10个神经元  输出层1个神经元

#先定义隐藏层 使用自带激励函数 tf.nn.relu
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# 定义输出层，此时输入为隐藏层的输出 l1, 输入10层，输出1层
prediction = add_layer(l1, 10, 1, activation_function=None)

#计算误差 方差  mean（平均值）
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
        # reduction_indices=[1]))
loss = tf.losses.mean_squared_error(ys,prediction)

# 优化，提高准确度，剃度下降法  0.1代表最小误差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 对系统初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 设计训练  让机器训练1000次，学习内容 train_step, 前面使用了 placeholder
# 所以后面要使用 feed_dict 字典来指定输入
# 使用 plt.figure()制作一下图, 表示用random生成的随机数据
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()# w使 plt.show() 不暂停，继续show
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # print(sess.run(loss,feed_dict={xs: x_data,ys: y_data}))
        try:
            # 为了让线的变化成为动画，需要不断的删除重新画
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'red',linewidth=5)
        plt.pause(0.1)

plt.ioff()
plt.show()




