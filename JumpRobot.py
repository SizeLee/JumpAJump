import tensorflow as tf
import numpy as np
import os
import sys
import subprocess
from PIL import Image
from scipy.misc import imread, imresize
import time
import random
import matplotlib.pyplot as plt

class JumpRobot:
    def __init__(self):
        self.swipe_x1 = 0
        self.swipe_y1 = 0
        self.swipe_x2 = 0
        self.swipe_y2 = 0
        self.state = None
        self.resize_state = None
        self.press_time = 300
        return

    def __screencap(self, filename):
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
        # self.__screencap('autojump.png')
        next_state = imread('autojump.png', mode='RGB')
        # print(type(next_state))
        # print(next_state.shape)
        self.state = next_state
        return next_state

    def decide_and_jump(self, jump_time, train_flag):
        # press_location_change_flag = True
        self.press_time = 400
        resize_width = 256

        for _ in range(jump_time):
            self.getNextState()
            ### here decide by neural network and basic judge if it's dead, then set the press time
            self.resize_state = imresize(self.state, size=(int(1920/1080*resize_width), resize_width))
            # plt.imshow(self.state)
            # plt.show()
            # plt.imshow(self.resize_state)
            # plt.show()

            die_flag = True ##decide by basic judge

            if die_flag:
                self.press_time = 800
                self.__set_button_position(self.state, die_flag)
                # press_location_change_flag = True
                die_flag = False


            # if press_location_change_flag:
            #     self.__set_button_position(self.state, die_flag)
            #     press_location_change_flag = False

            self.__press()

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
            ## todo here should find button position by picture
            h, w = im.shape[:2]
            left = int(w / 2)
            # top = int(1584 * (h / 1920.0))
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

    # def __del__(self):
    #     adb_path_str = '.\\adb\\'
    #     device_path_str = '/storage/emulated/0/DCIM/jumpshot/'
    #     os.system(adb_path_str + 'adb shell rm ' + device_path_str + '*.png')

if __name__ == '__main__':
    jump_robot = JumpRobot()
    # jump_robot.getNextState()
    jump_robot.decide_and_jump(1, True)
