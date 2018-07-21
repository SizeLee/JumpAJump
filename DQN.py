import tensorflow as tf
import numpy as np

class ZeroGamaDQN:
    def __init__(self, train_flag, imsize, weight_filename=None):
        self.train_flag = train_flag
        self.imsize = imsize
        ## 300ms~900ms, every 20ms is a decision
        self.decision_size = 31
        self.graph = tf.Graph()
        self.parameters = []
        self.session = tf.Session(graph=self.graph)
        self.__setup_net_structure()
        self.reset_weights()
        if weight_filename is not None:
            self.load_weights(weight_filename)
        return

    def reset_weights(self):
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            self.session.run(init)

    def __setup_net_structure(self):
        with self.graph.as_default():
            with tf.name_scope('input'):
                self.im = tf.placeholder(tf.float32, [None, self.imsize[0], self.imsize[1], self.imsize[2]], name='image')
            with tf.name_scope('conv1') as scope:
                out_channel = 32
                kernel = tf.Variable(tf.truncated_normal([8, 8, self.imsize[2], out_channel],
                                                         dtype=tf.float32,
                                                         stddev=1e-1),
                                     name='weights')
                conv = tf.nn.conv2d(self.im, kernel, [1, 4, 4, 1], padding='VALID')
                biases = tf.Variable(tf.constant(0.0, shape=[out_channel], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # with tf.name_scope('pool1'):
            #     self.pool1 = tf.nn.max_pool(self.conv1,
            #                                 ksize=[1, 2, 2, 1],
            #                                 strides=[1, 2, 2, 1],
            #                                 padding='VALID',
            #                                 name='pool1')

            with tf.name_scope('conv2') as scope:
                in_channel = 32
                out_channel = 16
                kernel = tf.Variable(tf.truncated_normal([4, 4, in_channel, out_channel],
                                                         dtype=tf.float32,
                                                         stddev=1e-1),
                                     name='weights')
                # conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='VALID')
                conv = tf.nn.conv2d(self.conv1, kernel, [1, 2, 2, 1], padding='VALID')
                biases = tf.Variable(tf.constant(0.0, shape=[out_channel], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # with tf.name_scope('pool2'):
            #     self.pool2 = tf.nn.max_pool(self.conv2,
            #                                 ksize=[1, 2, 2, 1],
            #                                 strides=[1, 2, 2, 1],
            #                                 padding='VALID',
            #                                 name='pool2')

            with tf.name_scope('conv3') as scope:
                in_channel = 16
                out_channel = 8
                kernel = tf.Variable(tf.truncated_normal([3, 3, in_channel, out_channel],
                                                         dtype=tf.float32,
                                                         stddev=1e-1),
                                     name='weights')
                # conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='VALID')
                conv = tf.nn.conv2d(self.conv2, kernel, [1, 1, 1, 1], padding='VALID')
                biases = tf.Variable(tf.constant(0.0, shape=[out_channel], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # with tf.name_scope('pool3'):
            #     self.pool3 = tf.nn.max_pool(self.conv3,
            #                                 ksize=[1, 2, 2, 1],
            #                                 strides=[1, 2, 2, 1],
            #                                 padding='VALID',
            #                                 name='pool3')

            with tf.name_scope('conv4') as scope:
                in_channel = 8
                out_channel = 4
                kernel = tf.Variable(tf.truncated_normal([3, 3, in_channel, out_channel],
                                                         dtype=tf.float32,
                                                         stddev=1e-1),
                                     name='weights')
                # conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='VALID')
                conv = tf.nn.conv2d(self.conv3, kernel, [1, 1, 1, 1], padding='VALID')
                biases = tf.Variable(tf.constant(0.0, shape=[out_channel], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # with tf.name_scope('pool4'):
            #     self.pool4 = tf.nn.max_pool(self.conv4,
            #                                 ksize=[1, 2, 2, 1],
            #                                 strides=[1, 2, 2, 1],
            #                                 padding='VALID',
            #                                 name='pool3')

            with tf.name_scope('fc1'):
                # shape = int(np.prod(self.pool4.get_shape()[1:]))
                shape = int(np.prod(self.conv4.get_shape()[1:]))
                # print(shape)
                # midsize = int(shape/3*2)
                midsize = int(shape / 4)
                w = tf.Variable(tf.truncated_normal([shape, midsize], dtype=tf.float32, stddev=1e-1), name='weight')
                b = tf.Variable(tf.constant(1.0, shape=[midsize], dtype=tf.float32), trainable=True, name='bias')
                # pool_flat = tf.reshape(self.pool4, [-1, shape])
                pool_flat = tf.reshape(self.conv4, [-1, shape])
                fc1 = tf.nn.bias_add(tf.matmul(pool_flat, w), b)
                self.fc1 = tf.nn.relu(fc1)
                self.parameters += [w, b]

            with tf.name_scope('fc2'):
                w = tf.Variable(tf.truncated_normal([midsize, self.decision_size], dtype=tf.float32, stddev=1e-1), name='weight')
                b = tf.Variable(tf.constant(1.0, shape=[self.decision_size], dtype=tf.float32), trainable=True, name='bias')
                fc2 = tf.nn.bias_add(tf.matmul(self.fc1, w), b)
                self.decision = tf.nn.softmax(fc2)
                self.parameters += [w, b]

            with tf.name_scope('train'):
                self.decision_label = tf.placeholder(tf.float32, shape=[None, self.decision_size], name='decision_label')
                self.reward_label = tf.placeholder(tf.float32, shape=[None, 1], name='reward_label')
                self.loss = tf.reduce_mean(
                                tf.square(
                                    tf.reduce_sum(self.decision_label * fc2, reduction_indices=1) - self.reward_label
                                )
                            )
                # self.loss = tf.reduce_mean(tf.square(self.decision - self.decision_label))
                # self.optimizer = tf.train.AdamOptimizer()
                self.optimizer = tf.train.AdadeltaOptimizer()
                # self.optimizer = tf.train.GradientDescentOptimizer(0.03)
                self.train_step = self.optimizer.minimize(self.loss)

            self.test_node = fc2
        return

    def run(self, input_state):
        decision_prob = self.session.run(self.decision, feed_dict={self.im: input_state})
        ## 300ms~900ms, every 20ms is a decision
        decision = np.argmax(decision_prob)
        press_time = 300 + decision*20
        # return press_time, decision
        return press_time, decision, decision_prob

    def train(self, input_state, decision_label, reward_label, train_degree):
        for i in range(train_degree):
            _, loss, test_node = self.session.run([self.train_step, self.loss, self.test_node],
                                                  feed_dict={self.im: input_state,
                                                             self.decision_label: decision_label,
                                                             self.reward_label: reward_label})
            # print('loss:', loss, test_node)
            print('loss:', loss)
        # print(test_node)
        return test_node

    def load_weights(self, weights_file_name):
        weights = np.load(weights_file_name)
        keys = sorted(weights.keys(), key=lambda x:int(x.split('_')[1]))
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            self.session.run(self.parameters[i].assign(weights[k]))
        return

    def save_weights(self, save_file_name):
        weights = []
        for each in self.parameters:
            weight = self.session.run(each)
            weights.append(weight)
        np.savez(save_file_name, *weights)
        return
