import tensorflow as tf
import numpy as np

#create data 生成100个随机数列 
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))#计算误差
#传播误差
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
#初始化
init = tf.initialize_all_variables()
### create tensorflow structure start ###

sess = tf.Session()
sess.run(init)#very important

#训练201步
for step in range(201):
    sess.run(train)
    #每格20步打印结果
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))


