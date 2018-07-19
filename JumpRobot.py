import os
from scipy.misc import *
import time
import random
import DQN
import matplotlib.pyplot as plt
import ScoreRecognition
import numpy as np

class JumpRobot:
    def __init__(self):
        self.swipe_x1 = 0
        self.swipe_y1 = 0
        self.swipe_x2 = 0
        self.swipe_y2 = 0
        self.state = None
        self.resize_state = None
        self.press_time = 300
        self.dqn = None
        self.last_decision = 0
        self.last_state = None
        self.last_d_prob = None
        self.score_recognizer = ScoreRecognition.ScoreRecognizer('./data/score_digit_samples/digit_weight.npz')
        return

    @staticmethod
    def __screencap(filename):
        adb_path_str = '.\\adb\\'
        device_path_str = '/storage/emulated/0/DCIM/jumpshot/'
        os.system(adb_path_str + 'adb shell rm ' + device_path_str + '*.png')
        while (True):
            os.system(adb_path_str + 'adb shell screencap -p ' + device_path_str + filename)
            flag = os.system(adb_path_str + 'adb pull ' + device_path_str + filename)
            if (flag == 0):
                break
        os.system(adb_path_str + 'adb shell rm ' + device_path_str + '*.png')
        # process = subprocess.Popen(adb_path_str + 'adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        # screenshot = process.stdout.read()
        # if sys.platform == 'win32':
        #     screenshot = screenshot.replace(b'\r\n', b'\n')
        # with open('autojump.png', 'wb') as f:
        #     f.write(screenshot)

    def getNextState(self):
        self.__screencap('autojump.png')
        next_state = imread('autojump.png', mode='RGB')
        # print(type(next_state))
        # print(next_state.shape)
        self.state = next_state
        return next_state

    @staticmethod
    def __rgb2grey(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return grey

    @staticmethod
    def __rgb_plus_grey(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
        mix = np.stack((r, g, b, grey), axis=2)
        return mix

    def __preprocess_state(self, state):
        resize_width = 128
        resize_height = int(1920 * 0.35 / 1080 * resize_width)
        decision_area_top = int(0.35 * 1920)
        decision_area_bottom = int(0.7 * 1920)
        # grey = self.__rgb2grey(state[decision_area_top:decision_area_bottom, :, :])
        mix = self.__rgb_plus_grey(state[decision_area_top:decision_area_bottom, :, :])
        # print(grey.shape)
        # plt.imshow(grey)
        # plt.show()
        # resize_state = imresize(grey, size=(resize_height, resize_width), interp='nearest')
        resize_state = imresize(mix, size=(resize_height, resize_width), interp='nearest')
        # plt.imshow(resize_state)
        # plt.show()
        return resize_state

    def decide_and_jump(self, jump_time, trainable_flag, save_flag, weights_file_name=None):
        # press_location_change_flag = True
        self.press_time = 400
        self.getNextState()
        resize_state = self.__preprocess_state(self.state)
        resize_width = resize_state.shape[0]
        resize_height = resize_state.shape[1]
        channel = resize_state.shape[2]
        self.dqn = DQN.ZeroGamaDQN(trainable_flag, (resize_height, resize_width, channel), weights_file_name)
        train_flag = False
        die_flag = False
        last_score = 0
        for _ in range(jump_time):
            print('trainging_round:', _)
            self.getNextState()
            ### here decide by neural network and basic judge if it's dead, then set the press time
            # self.resize_state = imresize(self.state, size=(resize_height, resize_width)).reshape((1, resize_height,
            #                                                                                       resize_width, 3))
            self.resize_state = self.__preprocess_state(self.state).reshape((1, resize_height, resize_width, channel))
            if die_flag:
                train_flag = False

            die_flag = self.__is_died(self.state) ##decide by basic judge

            if die_flag:
                if trainable_flag and train_flag:
                    train_degree = 6
                    # label = np.ones((1, self.dqn.decision_size))
                    # label = self.last_d_prob + self.last_d_prob[0, self.last_decision]/(self.dqn.decision_size - 1)
                    label = np.zeros((1, self.dqn.decision_size))
                    label[0, self.last_decision] = 1
                    reward = np.array([[-1]])
                    # label = label/(self.dqn.decision_size - 1)
                    # label = label / 2
                    self.dqn.train(self.last_state, label, reward, train_degree)

                self.press_time = 500
                # press_location_change_flag = True

            else:
                ### survive, so promote the last decision in last state
                cur_score = self.score_recognizer.recognize(self.state)
                if trainable_flag and train_flag:
                    ## todo set label and training degree by score change
                    train_degree = 5
                    reward = np.array([[(cur_score - last_score) % 10]])
                    print(reward)
                    label = np.zeros((1, self.dqn.decision_size))
                    label[0, self.last_decision] = 1
                    self.dqn.train(self.last_state, label, reward, train_degree)
                ##
                self.press_time, self.last_decision, self.last_d_prob = self.dqn.run(self.resize_state)
                self.last_state = self.resize_state
                if not train_flag:
                    train_flag = True
                last_score = cur_score

            # if press_location_change_flag:
            #     self.__set_button_position(self.state, die_flag)
            #     press_location_change_flag = False
            self.__set_button_position(self.state, die_flag)
            self.__press(die_flag)
            print('decision:', self.last_decision, 'press_time:', self.press_time)
            print()
            time.sleep(1)

        if save_flag:
            self.dqn.save_weights('autojump.npz')

        return

    def __read_samples(self):
        npzfile = np.load('./data/jump_sample/choice_rewards.npz')
        self.choice = npzfile['choice']
        self.rewards = npzfile['rewards']
        sample_num = len(self.choice)
        self.choice = self.choice.astype(np.int)
        # print(self.choice.dtype)
        # print(self.rewards)
        im = imread('./data/jump_sample/%d.png' % 0, mode='RGB')
        im = self.__preprocess_state(im)
        im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
        self.sample_states = im
        for i in range(1, sample_num):
            print(i)
            im = imread('./data/jump_sample/%d.png' % i, mode='RGB')
            im = self.__preprocess_state(im)
            im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
            self.sample_states = np.vstack((self.sample_states, im))
        self.sample_states = np.array(self.sample_states)
        print(self.sample_states.shape)
        return

    def train_by_records(self, train_epochs, mini_batch_size=8, save_file_name='./autojump_rec.npz'):
        self.__read_samples()
        sample_num = len(self.choice)
        self.rewards = self.rewards.reshape((sample_num, 1))
        self.dqn = DQN.ZeroGamaDQN(True, self.sample_states.shape[1:])
        choice_matrix = np.zeros((sample_num, self.dqn.decision_size))
        choice_matrix[[i for i in range(sample_num)], self.choice] = 1
        for _ in range(train_epochs):
            print('round %d' % _)
            mini_batch_index = random.sample([i for i in range(sample_num)], mini_batch_size)
            test_node = self.dqn.train(self.sample_states[mini_batch_index],
                           choice_matrix[mini_batch_index],
                           self.rewards[mini_batch_index],
                           1)
        print(test_node)
        self.dqn.save_weights(save_file_name)
        return

    def __press(self, die_flag):
        adb_path_str = '.\\adb\\'
        if die_flag:
            cmd = adb_path_str + 'adb shell input tap {x1} {y1}'.format(x1=self.swipe_x1, y1=self.swipe_y1)
        else:
            cmd = adb_path_str + 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
            x1=self.swipe_x1,
            y1=self.swipe_y1,
            x2=self.swipe_x2,
            y2=self.swipe_y2,
            duration=int(self.press_time)
        )
        os.system(cmd)
        return

    def __set_button_position(self, im, die_flag):
        if die_flag:
            """
            将 swipe 设置为 `再来一局` 按钮的位置
            """
            ## here should find button position by picture
            h, w = im.shape[:2]
            left = int(w / 2)
            if im[1, 1, 0] == im[-2, -2, 0] and im[1, 1, 1] == im[-2, -2, 1] and im[1, 1, 2] == im[-2, -2, 2]:
                top = int(1584 * (h / 1920.0))
            else:
                top = int(1431 * (h / 1920.0))
            left = int(random.uniform(left - 50, left + 230))
            top = int(random.uniform(top - 50, top + 50))  # 随机防 ban
            after_top = int(random.uniform(top - 50, top + 50))
            after_left = int(random.uniform(left - 50, left + 50))
            self.swipe_x1, self.swipe_y1, self.swipe_x2, self.swipe_y2 = left, top, after_left, after_top
        else:
            h, w = im.shape[:2]
            left = int(2/3 * w)
            top = int(2/3 * h)
            left = int(random.uniform(left - 100, left + 100))
            top = int(random.uniform(top - 100, top + 100))
            after_top = int(random.uniform(top - 50, top + 50))
            after_left = int(random.uniform(left - 50, left + 50))
            self.swipe_x1, self.swipe_y1, self.swipe_x2, self.swipe_y2 = left, top, after_left, after_top

        return

    @staticmethod
    def __is_died(state):
        ## judge whether the current state is died
        # [119 121 127] dead; [[215 219 230] alive
        if state[1][1][0] < 180 and state[1][1][1] < 180 and state[1][1][2] < 180:
            return True
        return False

    def test(self):
        x = imread('autojump.png', mode='RGB')
        g = self.__preprocess_state(x)
        plt.imshow(g, cmap='gray')
        plt.show()

if __name__ == '__main__':
    jump_robot = JumpRobot()
    # jump_robot.getNextState()
    # jump_robot.decide_and_jump(50, True, True)
    # jump_robot.decide_and_jump(200, True, True, 'autojump.npz')
    # jump_robot.test()
    jump_robot.train_by_records(100000, mini_batch_size=32)
    # jump_robot.decide_and_jump(50, False, False, 'autojump_rec.npz')

