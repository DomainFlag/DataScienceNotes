import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as fc

from torch.optim.rmsprop import RMSprop
from torch.tensor import Tensor
from typing import List, Callable


class DQN(nn.Module):

    size: int
    actions_count: int

    def __init__(self, size: int, actions_count: int):
        super(DQN, self).__init__()

        self.size, self.actions_count = size, actions_count

        self.sec1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.sec2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.sec3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.pipeline = [ self.sec1, self.sec2, self.sec3 ]
        for sec in self.pipeline:
            conv = next(sec.modules())

            size = (size + conv.padding * 2 - (conv.kernel_size - 1)) // conv.stride + 1

        self.ln = nn.Linear(size * size * conv.out_channels, actions_count)

    def forward(self, inputs):
        for sec in self.pipeline:
            inputs = sec.forward(inputs)

        return self.ln(inputs.view(inputs.size(0), -1))


class Transition:

    state: Tensor
    action: Tensor
    state_result: Tensor
    reward: Tensor

    def __init__(self, state: Tensor, action: Tensor, state_result: Tensor, reward: Tensor):
        self.state = state
        self.action = action
        self.state_result = state_result
        self.reward = reward


class ReplayMemory:

    capacity: int
    memory: List[Transition]
    position: int

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition: Transition):
        """ Save transition by possibly overriding pre-existing transitions """
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    size: int
    action_count: int
    memory_capacity: int
    rewarder: Callable
    step: int = 0

    def __init__(self, size, action_count, rewarder, memory_capacity):

        self.size, self.action_count, self.memory_capacity = size, action_count, memory_capacity
        self.rewarder = rewarder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(size, action_count).to(self.device)
        self.optimizer = RMSprop(self.model.parameters())
        self.memory = ReplayMemory(memory_capacity)

    def choose_action(self, state):
        sample = random.random()
        eps_threshold = Agent.EPS_END + (Agent.EPS_START - Agent.EPS_END) * np.exp(-1. * self.step / Agent.EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.model.forward(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_count)]], device = self.device, dtype = torch.long)
