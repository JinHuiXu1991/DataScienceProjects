from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import cv2
from matplotlib.pyplot import imshow
import sys
import time
import os
from datetime import datetime
from datetime import date
from game.flappy_bird import GameState
import math

BATCH_SIZE = 64
GAMMA = 0.99
#EPS_DECAY = 0.99999
EPS_START = 0.1
EPS_END = 0.0001
NUM_ITERATIONS = 2000000
REPLAY_MEMORY_SIZE = 10000
LEARNING_RATE = 1e-6
ACTION_NUM = 2

epsilon_decrements = np.linspace(EPS_START, EPS_END, NUM_ITERATIONS)

writer = SummaryWriter()

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
    
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)
        

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

class AgentBird:
    def __init__(self, action_num, model):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.exploration_rate = EPS_START
        self.model = model
        self.model = self.model.to(device)
        self.action_num = action_num
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = LEARNING_RATE)
        self.lossFuc = torch.nn.MSELoss()
        
    def return_model(self):
        return self.model
    
    def add_memory(self, state, action, reward, state_1, terminal):
        self.memory.append((state, action, reward, state_1, terminal))

    # predict agent action from input state
    def predict_action(self, state):
        # get q values for the state from DQN model
        q_values = self.model(state)[0]

        # apply greedy algorithm to select the action index
        if np.random.rand() < self.exploration_rate:
            action_index = random.randrange(self.action_num)
        else:
            action_index = torch.argmax(q_values).cpu().numpy().tolist()
        return action_index, q_values
    
    def experience_replay(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # only train when we have enough transitions for at least one batch
        if len(self.memory) < BATCH_SIZE:
            return 0
        
        # collect a batch of sample transitions from ReplayMemory
        batch = random.sample(self.memory, BATCH_SIZE)
        
        state_batch, action_batch, reward_batch, state_1_batch, terminal_batch = zip(*batch)
        
        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.cat(tuple(action for action in action_batch))
        reward_batch = torch.cat(tuple(reward for reward in reward_batch))
        reward_batch = reward_batch.view(len(reward_batch), 1)
        state_1_batch = torch.cat(tuple(state_1 for state_1 in state_1_batch))

        state_batch.to(device)
        action_batch.to(device)
        reward_batch.to(device)
        state_1_batch.to(device)
        
        # current state prediction
        current_prediction_batch = self.model(state_batch)
        # next state prediction
        next_prediction_batch = self.model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(
            tuple(reward if terminal else reward + GAMMA * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        # extract Q-value
        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        
        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()
        
        # Compute loss
        loss = self.lossFuc(q_value, y_batch)
        return_loss = loss.item()
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return return_loss
    
    def update_exploration_rate(self, steps_done):
        if steps_done < NUM_ITERATIONS:
            self.exploration_rate = epsilon_decrements[steps_done]
        else:
            self.exploration_rate = EPS_END
            
        # different strategies
        # EPS_DECAY = 0.99999
        # EPS_START = 0.9
        # EPS_END = 0.0001
        #self.exploration_rate *= EPS_DECAY
        #self.exploration_rate  = max(EPS_END, self.exploration_rate)

        
        #EPS_START = 0.9
        #EPS_END = 0.05
        #EPS_DECAY = 200 / 2000 (trial 3)
        #self.exploration_rate = EPS_END + (EPS_START - EPS_END) * \
        #math.exp(-1. * steps_done / EPS_DECAY)
        
        #self.exploration_rate = EPS_END + (EPS_START - EPS_END) * math.exp(-EPS_DECAY * steps_done)
        #self.exploration_rate = EPS_END + (
        #        (NUM_ITERATIONS - steps_done) * (EPS_START - EPS_END) / NUM_ITERATIONS)
        
def train(model):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                            
    # instantiate game
    game_state = GameState()
    agentBird = AgentBird(ACTION_NUM, model)
    # initial action is do nothing
    # [1, 0] represents "Do nothing"
    # [0, 1] represents "Fly up"
    action = torch.zeros([ACTION_NUM], dtype=torch.float32)
    action[0] = 1
    state_image, reward, terminal, state_score = game_state.frame_step(action)
    image = preprocess_image(state_image)
    state = torch.cat((image, image, image, image)).unsqueeze(0)

    
    run = 1
    i = 0
    while True:
    #for i in range(NUM_ITERATIONS):
        # Select and perform an action
        action_idx, q_values = agentBird.predict_action(state)
        action = torch.zeros([ACTION_NUM], dtype=torch.float32)
        action[action_idx] = 1
        
        # get next state and reward
        state_image_1, reward, terminal, state_score = game_state.frame_step(action)
        image_1 = preprocess_image(state_image_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_1)).unsqueeze(0)
        
        score = reward
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        reward = torch.tensor([reward])
        
        reward = reward.to(device)
        
        action = action.unsqueeze(0)
        action = action.to(device)
        
        # Store the transition in memory
        agentBird.add_memory(state, action, reward, state_1, terminal)
        agentBird.update_exploration_rate(i)
        
        # Perform one step of the optimization (on the target network)
        loss = agentBird.experience_replay()

        print ("iteration " + str(i) + ", exploration: " + str(agentBird.exploration_rate) + ", Q max:" + str(np.max(q_values.cpu().detach().numpy())) +
              ", action:" + str(action_idx) + ", reward:" + str(score))
        writer.add_scalar('Q_value', np.max(q_values.cpu().detach().numpy()), i)
        writer.add_scalar('loss', float(loss), i)
        
        # Move to the next state
        state = state_1
        i += 1
        
        if terminal:
            run += 1
            print ("episode " + str(run) + ", Score: " + str(state_score))
            writer.add_scalar('Score', state_score, run)

    
        if i % 50000 == 0:
            date_today = date.today()
            curr_time = datetime.now()
            formatted_time = curr_time.strftime('%H%M%S')
            save_model = agentBird.return_model()
            torch.save(save_model, "pretrained_model/easy_model_" + str(i) + "_" + str(date_today) + "_" + str(formatted_time) + ".pth")
        
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists('pretrained_model/'):
        os.mkdir('pretrained_model/')

    model = DQN()
    model.apply(init_weights)
    model.to(device)

    train(model)

        
if __name__ == "__main__":
    main()
