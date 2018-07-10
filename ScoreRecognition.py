import numpy as np
import tensorflow as tf
from scipy.misc import *
import matplotlib.pyplot as plt


class ScoreRecognizer:
    def __init__(self, weight_file_name = None):
        # self.score_area = (200, 290, 120, 280)
        self.resize_width = 128
        self.resize_height = int(1920/1080*self.resize_width)
        self.width_start_mark = int(120 / 1080 * self.resize_width) + 1
        self.height_start_mark = int(200/1920*self.resize_height) + 1
        self.digit_weight = 8
        self.digit_height = 10
        # here can set digit number by score_area, now it's three
        self.score_area = (self.height_start_mark, self.height_start_mark + self.digit_height,
                           self.width_start_mark, self.width_start_mark + 3 * self.digit_weight + 3)
        # im_sample= imresize(im_sample, (self.resize_height, self.resize_width), interp='nearest')
        # self.score_sample = im_sample[self.score_area[0]:self.score_area[1], self.score_area[2]:self.score_area[3]]
        # self.score_sample = self.__rgb2grey(self.score_sample)
        # print(self.score_sample.shape)
        self.digit_classifier = DigitRecognizer((self.digit_height, self.digit_weight), weight_file_name)
        # mid = self.score_sample[:, self.digit_weight + 1: self.digit_weight * 2 + 1]
        # plt.imshow(mid, cmap='gray')
        # plt.show()
        return

    def train(self, train_round):
        samples = []
        labels = []
        for i in range(10):
            file_name = './data/score_digit_samples/%d.png' % i
            im = imread(file_name)
            im = self.__resize(im)
            im = im[self.score_area[0]:self.score_area[1], self.score_area[2]:self.score_area[3]]
            im = self.__rgb2grey(im)
            im = self.__grey2bin(im)
            ## split digit
            digit_sample = im[:, :self.digit_weight]
            digit_label = np.zeros(11)
            digit_label[i] = 1
            samples.append(digit_sample)
            labels.append(digit_label)
            none_digit_sample = im[:, self.digit_weight + 1: self.digit_weight * 2 + 1]
            none_digit_label = np.zeros(11)
            none_digit_label[-1] = 1
            samples.append(none_digit_sample)
            labels.append(none_digit_label)

        self.digit_classifier.train(samples, labels, train_round)
        self.digit_classifier.save_weights('./data/score_digit_samples/digit_weight.npz')
        return

    def recognize(self, im):
        im = self.__resize(im)
        im = im[self.score_area[0]:self.score_area[1], self.score_area[2]:self.score_area[3]]
        im = self.__rgb2grey(im)
        im = self.__grey2bin(im)
        ## split digit and recognize
        take_left = 0
        take_right = take_left + self.digit_weight
        recognition_flag = False
        number = 0
        i = 0
        while(take_left < im.shape[1]):
            digit_im = im[:, take_left:take_right]
            digit = self.digit_classifier.run(digit_im)
            if digit is None:
                break
            else:
                recognition_flag = True
                number = number*10 + digit

            if i == 1:
                take_left += self.digit_weight + 2
                take_right += self.digit_weight + 2
            else:
                take_left += self.digit_weight + 1
                take_right += self.digit_weight + 1
            i += 1

        if recognition_flag:
            return number
        else:
            return None

    def __rgb2grey(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return grey

    def __grey2bin(self, grey):
        mean = np.mean(grey)
        binary = grey < mean
        return binary * 1

    def __resize(self, im):
        im = imresize(im, (self.resize_height, self.resize_width), interp='nearest')
        return im

    def test(self, im):
        im = self.__resize(im)
        im = im[self.score_area[0]:self.score_area[1], self.score_area[2]:self.score_area[3]]
        im = self.__rgb2grey(im)
        plt.imshow(im, cmap='gray')
        plt.show()
        im = self.__grey2bin(im)
        print(im)
        plt.imshow(im, cmap='binary')
        plt.show()
        return


class DigitRecognizer:
    def __init__(self, imsize, weight_file_name = None):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.parameters = []
        self.__init_digit_recognizer(imsize)
        if weight_file_name is not None:
            self.load_weights(weight_file_name)
        return

    def __init_digit_recognizer(self, imsize):
        with self.graph.as_default():
            with tf.name_scope('input'):
                self.im = tf.placeholder(tf.float32, shape=(None, *imsize), name='raw_input')
                feature_length = int(imsize[0]*imsize[1])
                flat_input = tf.reshape(self.im, shape=(-1, feature_length), name='reshape_input')
            with tf.name_scope('fc1') as scope:
                mid_layer_size = feature_length//2
                weight = tf.Variable(tf.truncated_normal((feature_length, mid_layer_size),
                                                         dtype=tf.float32, stddev=1e-1), name='weight')
                bias = tf.Variable(tf.constant(1.0, shape=[mid_layer_size], dtype=tf.float32), name='bias')
                fc1_out = tf.nn.bias_add(tf.matmul(flat_input, weight), bias)
                self.fc1 = tf.nn.relu(fc1_out, name=scope)
                self.parameters += [weight, bias]

            with tf.name_scope('fc2') as scope:
                digit_type_num = 11 ## 0-9 and none
                weight = tf.Variable(tf.truncated_normal((mid_layer_size, digit_type_num),
                                                         dtype=tf.float32, stddev=1e-1), name='weight')
                bias = tf.Variable(tf.constant(1.0, shape=[digit_type_num], dtype=tf.float32), name='bias')
                self.logits = tf.nn.bias_add(tf.matmul(self.fc1, weight), bias)
                self.fc2 = tf.nn.softmax(self.logits, name=scope)
                self.parameters += [weight, bias]

            with tf.name_scope('train'):
                self.label = tf.placeholder(tf.float32, (None, digit_type_num), name='label')
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                   labels=self.label))
                optimizer = tf.train.AdamOptimizer()
                self.train_step = optimizer.minimize(self.loss)
                self.accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(self.fc2, axis=1), tf.argmax(self.label, axis=1)),
                    tf.float32))

            # print(self.im.shape)
            self.varibles_init = tf.global_variables_initializer()
            self.sess.run(self.varibles_init)
        return

    def load_weights(self, weights_file_name):
        weights = np.load(weights_file_name)
        keys = sorted(weights.keys(), key=lambda x:int(x.split('_')[1]))
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            self.sess.run(self.parameters[i].assign(weights[k]))
        return

    def save_weights(self, save_file_name):
        weights = []
        for each in self.parameters:
            weight = self.sess.run(each)
            weights.append(weight)
        np.savez(save_file_name, *weights)
        return

    def train(self, samples, labels, train_epoches):
        for i in range(train_epoches):
            loss, accuracy, _ = self.sess.run([self.loss, self.accuracy, self.train_step],
                                              feed_dict={self.im: samples, self.label: labels})
            print(i, loss, accuracy)
        return

    def run(self, sample):
        if len(sample.shape) == 2:
            sample = [sample]
        kind_prob = self.sess.run(self.fc2, feed_dict={self.im: sample})
        kind = np.argmax(kind_prob)
        if kind == 10:
            return None
        return kind

if __name__ == '__main__':
    # im = imread('./data/score_digit_samples/145.png')
    # plt.imshow(im)
    # plt.show()
    # sr = ScoreRecognizer()
    # sr.test(im)
    # sr.train(100)
    sr = ScoreRecognizer('./data/score_digit_samples/digit_weight.npz')
    im = imread('./data/score_digit_samples/107.png')
    # sr.test(im)
    re = sr.recognize(im)
    print(re)


