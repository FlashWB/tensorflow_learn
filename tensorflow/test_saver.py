import tensorflow as tf
import numpy as np
## Save to file
# remember to define the same dtype and shape when restore
# W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weight')
# b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

# init = tf.global_variables_initializer()

# saver = tf.train.Saver() # 建立一个tf.train.Saver()用来保存

# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess,"net_/save_net.ckpt")
#     print("Save to path:", save_path)


## 提取 saver.restore()
W = tf.Variable(np.arange(6).reshape(2,3), dtype=tf.float32, name="weight")
b = tf.Variable(np.arange(3).reshape(1,3), dtype=tf.float32, name="biases")

# 不需要 init

saver = tf.train.Saver()
with tf.Session() as sess:
    # 提取变量
    saver.restore(sess,"net_/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))





