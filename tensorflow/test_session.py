###  Session()的两种打开方法 效果相同 ##
# 1 不使用with
# 2 使用with
import tensorflow as tf

maxtrix1 = tf.constant([[3,3]])
maxtrix2 = tf.constant([[2],[2]])

product = tf.matmul(maxtrix1,maxtrix2)  #matrix multiply np.dot(m1,m2)

# method 1 
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# methord 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)














