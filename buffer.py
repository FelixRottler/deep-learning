import numpy as np
import torch

class ReplayBuffer():

    def __init__(self, input_shape, memory_capacity, batch_size=8) -> None:
        self.memory_capacity = memory_capacity
        self.state_memory = torch.zeros(self.memory_capacity, *input_shape, dtype=torch.float32) 
        self.action_memory = torch.zeros(self.memory_capacity, dtype=torch.int16)
        self.reward_memory = torch.zeros(self.memory_capacity, 4)
        self.next_state_memory = torch.zeros(self.memory_capacity, *input_shape, dtype=torch.float32)
        self.terminal_memory = torch.zeros(self.memory_capacity, dtype=torch.bool)

        self.memory_index = 0
        self.batch_size = batch_size

    def store_experience(self, state, action, reward, next_state, terminal):
        if self.memory_index == self.memory_capacity:
            self.memory_index = 0
        # print("    STORING EXPERIENCE")
        self.state_memory[self.memory_index] = state
        self.action_memory[self.memory_index] = action
        self.reward_memory[self.memory_index] = reward
        self.next_state_memory[self.memory_index] = next_state
        self.terminal_memory[self.memory_index] = terminal

        self.memory_index += 1

    def sample_minibatch_from_memory(self):
        mask = np.random.choice(self.memory_capacity , size = self.batch_size)

        state_batch = self.state_memory[mask]
        next_state_batch = self.next_state_memory[mask]
        reward_batch = self.reward_memory[mask]
        action_batch = self.action_memory[mask]
        terminal_batch = self.terminal_memory[mask]
        
        return state_batch, next_state_batch, reward_batch, action_batch, terminal_batch

