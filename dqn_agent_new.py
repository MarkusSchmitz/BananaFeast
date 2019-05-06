import numpy as np 
import random
from collections import namedtuple, deque
from model import QNetwork
from model import QNetwork2
from memory import Memory
import torch
import torch.nn.functional as F
import torch.optim as optim

#Check for GPU, enable if exists
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    '''DQN Agent that learns from the environment'''

    def __init__(self, state_size, action_size, seed, 
        BUFFER_SIZE = int(1e5), BATCH_SIZE = 64, GAMMA = 0.99, 
        TAU = 1e-3, LR = 5e-4, UPDATE_INTERVAL = 20, EPSILON = 1, model=True):

        '''
        Initialize Agent
        Params:
        -----------------------------------------------
        state_size          # size of state space
        action_size         # size of action space
        seed                # random seed for reproduceability
        BUFFER_SIZE         # replay buffer size
        BATCH_SIZE          # minibatch size
        GAMMA               # discount factor
        TAU                 # for soft update of target parameters
        LR                  # learning rate 
        UPDATE_INTERVAL     # how often to update the network
        EPSILON             # Exploration Rate
        -----------------------------------------------
        '''
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE       
        self.GAMMA = GAMMA         
        self.TAU = TAU            
        self.LR = LR              
        self.UPDATE_INTERVAL = UPDATE_INTERVAL  
        self.EPSILON = EPSILON 
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.exp = 0
        self.lvl = 0
        self.scoremax = 0

        # init Qnetworks
        if model == True:
            self.localNet = QNetwork(state_size, action_size, seed).to(device)
            self.targetNet = QNetwork(state_size, action_size, seed).to(device)
        else:
            self.localNet = QNetwork2(state_size, action_size, seed).to(device)
            self.targetNet = QNetwork2(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.localNet.parameters(), lr=LR)

        # init experience memory
        self.memory = Memory(BUFFER_SIZE, BATCH_SIZE)

        # timestep for timetracking
        self.timestep = 0


    def step(self, state, action, reward, next_state, done):
        '''
        manage experience and learning for each timestep
        '''
        # add Experience to Memory
        self.memory.add_experience(state,action,reward,next_state, done)
        self.exp +=1 

        # updates the timestep for each step. 
        # resets each training interval.
        self.timestep = (self.timestep + 1) % self.UPDATE_INTERVAL

        # Update the net each time an interval has finished
        if (self.timestep == 0): # finished one interval
            if (len(self.memory.memory) > self.BATCH_SIZE):
                experiences = self.memory.sample_Batch()
                self.learn(experiences)
    
    def get_action(self, state):
        '''
        return the action recommended by the Algorithm for the given state

        Params:
        state: State of the environment for calculating Action
        '''
        # Check if epsilon --> exploration
        if self.EPSILON > random.random():
            # if exploration return random action
            return random.choice(np.arange(self.action_size))
        else: # Calculate Greedy Action

            # get state and transform into network readable tensor
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

            # set Network to evaluation Mode (disables modification of Net)
            self.localNet.eval()

            # Get Action Values by passing state to the local Network
            with torch.no_grad():
                action_values = self.localNet(state)
            
            #Reset to training (modification) Mode
            self.localNet.train()
            
            # return most valuable (greedy) action
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences):
        '''
        Update localNet and targetNet through experience
        
        Params:
        experiences: random sample of SARS from Memory
        '''

        # Unpack Values saved in Experience as Tuples
        states, actions, rewards, next_states, dones = experiences

        # Get max Q Values of next states from target model
        qvals_next = self.targetNet(next_states).detach().max(1)[0].unsqueeze(1)
        # Calculate expected Rewards based on current values, make 0 if done
        qtargets = rewards + (self.GAMMA * qvals_next * (1 - dones))
        # Get expected Q Values from Local Model
        qvals_expected = self.localNet(states).gather(1, actions)

        # compute loss with mean squared error
        loss = F.mse_loss(qvals_expected, qtargets)
        # Reset the gradients
        self.optimizer.zero_grad()
        #  Use Optimizer to gradient descent on weights
        loss.backward()
        self.optimizer.step()

        # update target Network
        self.update_targetNet(self.localNet, self.targetNet)
        self.lvl += 1

    def update_targetNet(self, local_model, target_model):
        '''Update the targetNet dpeending on localNet
        
        Params:
        local_model: local model trained on exp 
        
        # iterate through all layers and weights
        for local_weight, target_weight in zip(target_model.parameters(), local_model.parameters()):
            
            # calculate the new weights
            updated_weight = self.TAU * local_weight.data + (1.0 - self.TAU) * target_weight.data

            #write new weights to targetNet
            target_weight.data.copy_(updated_weight)
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

    def get_stats(self):
        '''
        returns stats of Agent
        '''

        return (self.exp, self.lvl // 100 , self.scoremax)