#coding: utf-8
import tensorflow as tf
import numpy as np

#生成0和1矩阵
v1 = tf.Variable(tf.zeros([3,3,3]),name="v2")
v2 = tf.Variable(tf.ones([10,5]),name='v2')

#填充单值矩阵
v3 = tf.Variable(tf.fill([2,3],9))

#常量矩阵
v4_1 = tf.constant([1, 2, 3, 4, 5, 6, 7])
v4_2 = tf.constant(-1.0, shape=[2,3])

#生成等着数列
v6_1 = tf.linspace(10.0, 12.0, 30, name="linspace")


#初始化
Init_op = tf.initialize_all_variables()

#保存变量，可以指定保存的内容
# saver = tf.train.Saver({"my_v2": v2})
Saver = tf.train.Saver()

#运行
with tf.Session() as sess:
    sess.run(Init_op)
    #输出形状和值
    print (tf.Variable.get_shape(v1))
    print (sess.run(v1))

    #numpy保存文件
    np.save("v1.npy",sess.run(v1))
    test_a = np.load("v1.npy")
    print (test_a[1,2])

















