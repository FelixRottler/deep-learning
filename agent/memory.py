import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, input_shape, memory_capacity, batch_size=8) -> None:
        self.memory_capacity = memory_capacity

        self.state_memory = np.zeros(
            (self.memory_capacity, *input_shape), dtype=np.float32
        )
        self.action_memory = np.zeros(self.memory_capacity, dtype=np.int64)
        self.reward_memory = np.zeros(self.memory_capacity, dtype=np.float32)
        self.next_state_memory = np.zeros(
            (self.memory_capacity, *input_shape), dtype=np.float32
        )
        self.terminal_memory = np.zeros(self.memory_capacity, dtype=np.bool)

        self.memory_index = 0
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, terminal):

        self.memory_index = self.memory_index % self.memory_capacity
        self.state_memory[self.memory_index] = state
        self.action_memory[self.memory_index] = action
        self.reward_memory[self.memory_index] = reward
        self.next_state_memory[self.memory_index] = next_state
        self.terminal_memory[self.memory_index] = terminal

        self.memory_index += 1

    def sample(self):
        max_mem = min(self.memory_index, self.memory_capacity)
        mask = np.random.choice(max_mem, size=self.batch_size)

        state_batch = torch.from_numpy(self.state_memory[mask]).float()
        next_state_batch = torch.from_numpy(self.next_state_memory[mask]).float()
        reward_batch = torch.from_numpy(self.reward_memory[mask])
        action_batch = torch.from_numpy(self.action_memory[mask])
        terminal_batch = torch.from_numpy(self.terminal_memory[mask])
        return state_batch, next_state_batch, reward_batch, action_batch, terminal_batch
