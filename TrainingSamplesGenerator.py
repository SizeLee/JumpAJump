import time
import os
import random
from scipy.misc import *
import ScoreRecognition
import numpy as np

class TrainingSamplesGenerator:
    def __init__(self):
        self.swipe_x1 = 0
        self.swipe_y1 = 0
        self.swipe_x2 = 0
        self.swipe_y2 = 0
        self.state = None
        self.resize_state = None
        self.press_time = 300
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

    def __preprocess_state(self, state):
        resize_width = 128
        resize_height = int(1920 * 0.35 / 1080 * resize_width)
        decision_area_top = int(0.35 * 1920)
        decision_area_bottom = int(0.7 * 1920)
        grey = self.__rgb2grey(state[decision_area_top:decision_area_bottom, :, :])
        # print(grey.shape)
        # plt.imshow(grey)
        # plt.show()
        resize_state = imresize(grey, size=(resize_height, resize_width), interp='nearest')
        # plt.imshow(resize_state)
        # plt.show()
        return resize_state

    def __press(self):
        adb_path_str = '.\\adb\\'
        cmd = adb_path_str + 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
            x1=self.swipe_x1,
            y1=self.swipe_y1,
            x2=self.swipe_x2,
            y2=self.swipe_y2,
            duration=self.press_time
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
            left = int(random.uniform(left - 100, left + 230))
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

    ##todo random pick jump time by gaussian distribution
    def generate_samples_by_random(self, sample_number):
        rand_jump_time = np.random.normal(loc=540, scale=150, size=sample_number)
        # print(rand_jump_time)
        rand_jump_time = np.ceil(rand_jump_time/20) * 20
        reserve_location = rand_jump_time >= 200
        change_location = rand_jump_time < 200
        rand_jump_time = reserve_location * rand_jump_time + 200 * change_location
        reserve_location = rand_jump_time <= 800
        change_location = rand_jump_time > 800
        rand_jump_time = reserve_location * rand_jump_time + 800 * change_location
        # print(rand_jump_time)
        choice = (rand_jump_time-200)/20
        # print(choice)
        die_flag = False
        last_score = 0
        gotten_sample = 0
        record_flag = False
        last_state = None
        samples_state = []
        rewards = []
        while(True):
            state = self.getNextState()
            die_flag = self.__is_died(state)
            if die_flag:
                self.__set_button_position(state, die_flag)
                self.press_time = 300
                self.__press()
                if record_flag:
                    samples_state.append(last_state)
                    rewards.append(-2)
                    gotten_sample += 1
                    record_flag = False

            else:
                cur_score = self.score_recognizer.recognize(state)
                reward = (cur_score - last_score) % 10
                if record_flag:
                    samples_state.append(last_state)
                    rewards.append(reward)
                    gotten_sample += 1
                else:
                    record_flag = True
                last_state = state
                last_score = cur_score
                self.press_time = rand_jump_time[gotten_sample]
                self.__set_button_position(state, die_flag)
                self.__press()

            if gotten_sample >= sample_number:
                break
            time.sleep(1)
        # todo save all the picture state and other data in file

        return

if __name__ == '__main__':
    tsg = TrainingSamplesGenerator()
    tsg.generate_samples_by_random(100)