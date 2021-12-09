import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import cv2

import sys

import os

from game.flappy_bird_sound import GameState
import math
import argparse

ACTION_NUM = 2

class DQN(nn.Module):

    def __init__(self):
        '''
        output size: (((W - K + 2P)/S) + 1)
        Here W = Input size
        K = Filter size
        S = Stride
        P = Padding 
        '''
        super(DQN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))
        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
def preprocess_image(image):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.flip(image,1)
    
    # remove ground
    image = image[:400,:]
    
    # resize image
    image = cv2.resize(image, (84, 84))
    
    # convert image to grayscale
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    retval, image = cv2.threshold(image, 158, 255, cv2.THRESH_BINARY)
    
    # convert image data to desire tensor shape
    image = torch.FloatTensor(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    
    return image

class MasterBird:
    def __init__(self, action_num, model):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model
        self.model = self.model.to(device)
        self.action_num = action_num

    # demo predict function
    def play_action(self, state):
        # get q values for the state from DQN model
        q_values = self.model(state)[0]
        action_index = torch.argmax(q_values).cpu().numpy().tolist()
        return action_index, q_values
    

def demo(model, mode):
    # instantiate game
    game_state = GameState(mode)
    masterBird = MasterBird(ACTION_NUM, model)

    # initial action is do nothing
    action = torch.zeros([ACTION_NUM], dtype=torch.float32)
    action[0] = 1
    state_image, reward, terminal, state_score = game_state.frame_step(action, mode)
    image = preprocess_image(state_image)
    state = torch.cat((image, image, image, image)).unsqueeze(0)

    while True:
        action_idx, q_values = masterBird.play_action(state)
        action = torch.zeros([ACTION_NUM], dtype=torch.float32)
        action[action_idx] = 1
        
        # get next state
        state_image_1, reward, terminal, state_score = game_state.frame_step(action, mode)
        image_1 = preprocess_image(state_image_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_1)).unsqueeze(0)
        
        # set state to be state_1
        state = state_1
        
        if terminal:
            print ("Score: " + str(state_score))
         
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flappy Bird Demo')
    parser.add_argument('-m','--mode', help='Game Mode (easy/hard/master)', required=True)
    args = vars(parser.parse_args())
    mode = args['mode']
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if mode == 'easy':
        model = torch.load('pretrained_model/color_ez_model_4000000_2021-11-23_181113.pth', map_location='cpu').eval()
        model.to(device)
        demo(model, mode)
    elif mode == 'hard':
        model = torch.load('pretrained_model/hard_model_5000000_2021-11-24_150221.pth', map_location='cpu').eval()
        model.to(device)
        demo(model, mode)
    elif mode == 'master':
        model = torch.load('pretrained_model/master_model_5000000_2021-11-24_123516.pth', map_location='cpu').eval()
        model.to(device)
        demo(model, mode)
