
from typing import ForwardRef
from numpy.random import gamma
from pygame.surfarray import array2d
from game import  Game
from buffer import ReplayBuffer
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

size=(320,240)

class ManualAgent():
    def next_move(self, game):
        for event in pygame.event.get():
            return self.process_input(event)
    
    def process_input(self,event):
        if event.type == pygame.KEYDOWN:
            #up=1
            #left=2
            #right=3
            #down =4   
            if event.key == pygame.K_w and self.snake.direction != 4:     
                return 1
            elif event.key == pygame.K_a and self.snake.direction != 3:     
                return 2
            elif event.key == pygame.K_d and self.snake.direction != 2:    
                return 3
            elif event.key == pygame.K_s and self.snake.direction != 1:     
                return 4
    def receive_feedback(self,grow, game_over):
        pass 
        
class RandomAgent():
    def next_move(self, game):
        return np.random.randint(1,5)
    def receive_feedback(self,grow, game_over):
        pass 

class DQNNetwork(nn.Module):
    def __init__(self, input_shape,fc1_dim, fc2_dim, number_actions, batch_size) -> None:
        super(DQNNetwork,self).__init__()
        self.batch_size= batch_size
        self.input_shape=input_shape
        self.net = nn.Sequential(   
                nn.Linear(input_shape,fc1_dim),
                nn.ReLU(),
                nn.Linear(fc1_dim,fc2_dim),
                nn.ReLU(), 
                nn.Linear(fc2_dim, number_actions)
        )

        self.criterion = torch.nn.MSELoss()
    def forward(self,observations):
        # print("        OBSERVATIONS")
        # print("        "+str(observations.dtype))
        # observations dimension --> tensor(batch_size,3,32,32)
        observations = observations.view(observations.shape[0],-1) # 3*32*32
        #dimesion (5,3*32*32)
        # print("        "+str(observations.shape))
        return self.net(observations)
   

class DQNAgent():
    def __init__(self, model,input_shape ,epsilon,memory_capacity,gamma,episodes=50,number_actions=4,batch_size=8) -> None:
        self.model = model # finding the optimal q-function

        self.epsilon = epsilon
        self.gamma = gamma

        # experience replay
        # agent L --> execute L --> 
        self.memory_capacity = memory_capacity

        self.episodes = episodes
        self.number_actions = number_actions
        self.batch_size= batch_size
        self.buffer = ReplayBuffer(input_shape, memory_capacity, batch_size)

    def choose_next_action(self, state):
        if np.random.rand() > self.epsilon:
            # Exploration Phase
            action = np.random.choice(self.number_actions)
        else:
            # Exploitation Phase
            action = torch.argmax(self.model(state)).item()
        return action

    def learn(self, game):
        while not game.game_over:
            state = (torch.tensor(game.get_state(), dtype=torch.float32) / 255.0).unsqueeze(0)
            action = self.choose_next_action(state) 

            reward, next_state, terminal = game.execute_action(action) 
            reward_vector = torch.zeros(4)
            reward_vector[action] = reward
            self.buffer.store_experience(state, action, reward_vector, torch.tensor(next_state), terminal)

            if self.buffer.memory_index < self.batch_size:
                #print("NOT ENOUG MEMORY ----- SKIPPING")
                continue
            
            #print("LEARNING FROM MEMORY")
            state_batch, next_state_batch, reward_batch, action_batch, terminal_batch = self.buffer.sample_minibatch_from_memory()
            mask_terminal = np.where(terminal_batch == True)
            mask_non_terminal = np.where(terminal_batch == True)

            target = torch.zeros(self.batch_size, 4, dtype=torch.float32)
            target[mask_terminal] = reward_batch[mask_terminal]
            target[mask_non_terminal] = reward_batch[mask_non_terminal] + self.gamma * torch.argmax(self.model(next_state_batch)).item()
            target.requires_grad = True

            loss = self.model.criterion(self.model(state_batch), target)
            loss.backward()
        



epochs = 100

size=(64,64)
dqn_agent = DQNAgent(DQNNetwork(64*64,256, 256, 4, 8),size, 0.5,5000,0.8)
game = Game(size)

for i in range(epochs):
    dqn_agent.learn(game)
    game.reset()



                
                

