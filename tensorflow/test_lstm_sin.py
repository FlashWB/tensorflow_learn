# https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf20_RNN2.2/full_code.py
# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-09-RNN3/
# 使用 sin函数预测 cos函数

import tensorflow as tf
import numpy as np
import improtlib.pyplot as plt

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50  #这个是什么 
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006

# 定义生成数据的函数
def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape = (50batchs, 20steps) 生成 50 * 20 个数据，并reshape成矩阵
    xs = np.arange(BATCH_START, BATCH_START + BATCH_SIZE*TIME_STEPS).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    return [seq[:,:,np.newaxis], res[:,:,np.newaxis], xs]

# 定义 LSTM 的主体
class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size  # 一个时间点上的输出
        self.cell_size = cell_size
        self.batch_size = batch_size  # 一次传输给神经网络的数据
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32,[None, n_steps,input_size], name='xs')
            self.ys = tf.placeholder(tf.float32,[None, n_steps,output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

# 设置 add_input_layer, 添加 input_layer:
def add_input_layer(self,):
    l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D') # 一维(batch*n_step, in_size)
    # Ws(in_size, cell_size)
    Ws_in = self._weight_variable([self.input_size, self.cell_size])
    # bs(cell_size,)
    bs_in = self._bias_variable([self.cell_size,])
    # l_in_y = (batch * n_steps, cell_size)
    with tf.name_scope('Wx_plus_b'):
        l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
    # reshae l_in_y  ==> (batch, n_steps, cell_size)
    self.l_in_y = tf.reshape(l_in_y,[-1, self.n_steps, self.cell_size], name='2_3D')

# http://blog.csdn.net/u012436149/article/details/52887091 有一些简单介绍
# 添加cell 
def add_cell(self):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
    with tf.name_scope('initial_state'):
        # 生成初始化网络的state
        self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype = tf.float32)
    self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
        lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False
    )

# 添加output_layer
def add_output_layer(self):
    # shape = (batch * steps, cell_size)
    l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
    Ws_out = self._weight_variable([self.cell_size, self.output_size])
    bs_out = self._bias_variable([self.output_size,])
    # shape = (batch * steps, output_size)
    with tf.name_scope('Wx_plus_b'):
        self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

# 剩余计算
def compute_cost(self):
    losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [tf.reshape(self.pred, [-1], name='reshape_pred')],
        [tf.reshape(self.ys, [-1], name='resape_target')],
        [tf.reshape([self.batch_size * self.n_steps], dtype =tf.float32)],
        average_across_timesteps=ture,
        softmax_loss_function = self.ms_error,
        name = 'losses'
    )
    with tf.name_scope('average_cost'):
        self.cost = tf.div(
            tf.reduce_sum(losses, name'losses_sum),
            self.batch_size,
            name='averaget_cost'
        )
    tf.summary.scalar('cost', self.cost)

def ms_errro(labels, logits):
    return tf.square(tf.subtract(labels, logits))

def _weight_variable(self, shape, name='weight'):
    initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

def _bias_variable(self, shape, name='biases'):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable(name=name, shape=shape, initializer=initializer)

if __name__ == '__name__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)

    init = tf.global_variable_initializer()
    sess.run(init)

    plt.ion()
    plt.show()
    for i in range(200):
       seq, res, xs = get_batch()  #get_batch()获得数据
       if i == 0:
           feed_dict = {
               # initial state
               model.xs: seq,
               model.ys: res,
           }
        else:
            feed_dict = {
                # use last state as the initial state for this run
                model.xs: seq,
                model.ys: res,
                model.cell_init_state
            }





