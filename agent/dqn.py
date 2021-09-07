from agent.memory import ReplayBuffer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np


class DQNNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        action_dim,
        batch_size,
        num_filters=[32, 64, 64],
        filter_sizes=[(6,6), (4, 4), (3, 3)],
        fc_hidden_dim=512,
    ) -> None:
        super(DQNNetwork, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        # 80x80 --> 6x6 stride 2 ---> 38x38
        self.conv1 = nn.Conv2d(input_dim[0], num_filters[0], filter_sizes[0], stride=2)
        # 38x38 --> 4x4 stride 2 --> 18x18
        self.conv2 = torch.nn.Conv2d(
            num_filters[0], num_filters[1], filter_sizes[1], stride=2
        )
        # 18x18 --> 3x3 --> 16
        self.conv3 = torch.nn.Conv2d(
            num_filters[1], num_filters[2], filter_sizes[2], stride=1
        )
        self.fc1 = torch.nn.Linear(num_filters[2] * 16 * 16, fc_hidden_dim)
        self.fc2 = torch.nn.Linear(fc_hidden_dim, action_dim)

        self.criterion = torch.nn.SmoothL1Loss()

    def forward(self, state):
    
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(state.shape[0], -1)))
        return self.fc2(x)


class Resize(object):
    def __init__(self, factor):
       self.factor =factor
    def __call__(self, sample):
        return sample[:,::self.factor,::self.factor]
class ToTensor(object):
    
    def __call__(self,sample):
        return torch.tensor(sample/ 255.)
class DQNAgent:
    def __init__(
        self,
        input_shape,
        epsilon,
        memory_capacity,
        gamma,
        number_actions=4,
        batch_size=32,
        min_epsilon=0.1,
        epsilon_decay=0.98,
        lr=0.00025,
        transform =None
    ) -> None:

        self.q_eval = DQNNetwork(
            input_dim=input_shape, action_dim=number_actions, batch_size=batch_size
        )  # finding the optimal q-function
        self.q_target = DQNNetwork(
            input_dim=input_shape, action_dim=number_actions, batch_size=batch_size
        )  # finding the optimal q-function
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay

        self.lr = lr
        self.optim = torch.optim.Adam(self.q_eval.parameters(), lr=self.lr)

        # experience replay
        # agent L --> execute L -->
        self.memory_capacity = memory_capacity

        self.number_actions = number_actions
        self.batch_size = batch_size
        self.memory = ReplayBuffer(input_shape, memory_capacity, batch_size)

        self.transform = transform


    def process(self, observation):
        return self.transform(observation).unsqueeze(0).float()

    def choose_next_action(self, state):
        if np.random.rand() < self.epsilon:
            # Exploration Phase
            action = np.random.choice(self.number_actions)
        else:
            # Exploitation Phase
            with torch.no_grad():
                action = torch.argmax(self.q_eval(state)).item()
        return action

    def update_target_network(self):
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def store_transition(self, observation, action, reward, next_observation, done):
        self.memory.push(observation, action, reward, next_observation, done)

    def learn(self):
        if self.memory.memory_index < self.batch_size:
            return
        self.optim.zero_grad()

        batch_index = torch.arange(self.batch_size, dtype=torch.int64)
        (
            state_batch,
            next_state_batch,
            reward_batch,
            action_batch,
            terminal_batch,
        ) = self.memory.sample()
        

        predictions = self.q_eval.forward(state_batch)[batch_index, action_batch]
        target_actions = torch.argmax(self.q_eval.forward(next_state_batch),dim=1)
        # Predict the probabilities of which action we would take i the next step
        with torch.no_grad():
            next_predictions = self.q_target.forward(next_state_batch)[batch_index,target_actions]
           
        next_predictions[terminal_batch] = 0.0
        
        # Extract the maximum of the q value of the action that we would take in the next timestep
        # This means ok I am the snake and for this given state I would predict that I would go left
        # This is what we EXPECT --> next time we would expect that the snake goes left
        target = reward_batch + self.gamma * next_predictions
        
        # So we check if we actually would go left
        # For example
        # NO right we would go right but in the next step we would go left so there is a big error and this is our loss
        loss = self.q_eval.criterion(target, predictions)
        loss.backward()

        self.optim.step()

    def save_model(self, path):
        torch.save(self.q_eval.state_dict(), path)
