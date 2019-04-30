import numpy as np 
import random
from collections import namedtuple, deque
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
'''Constants'''

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 20        # how often to update the network

#Check for GPU, enable if exists
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    '''DQN Agent that learns from the environment'''

    def __init__(self, state_size, action_size, seed, ):
        '''
        Initialize Agent
        Params:
        -----------------------------------------------
        



        -----------------------------------------------
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # init Qnetworks
        self.localNet = QNetwork(state_size, action_size, seed).to(device)
        self.targetNet = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # init experience memory
        self.memory = Memory(action_size, state_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # timestep for timetracking
        self.timestep = 0


        def step(self, state, action, reward, next_state, done):

            return

        def get_action():

            return

        def learn():
            return

        def update_targetNet():
            return

class Memory(self, action_size, state_size, BUFFER_SIZE, BATCH_SIZE, seed):

    def __init__():
        return

    def add_experience():

        return
    def sample_Batch():
        return

    