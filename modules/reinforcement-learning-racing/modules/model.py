import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as fc

from torch.optim.rmsprop import RMSprop
from torch.tensor import Tensor
from typing import List, Callable, Optional, Any


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

    bagging: float = 0.34
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

    def sample(self, batch_size) -> Optional[List[Transition]]:
        if np.random.random() < ReplayMemory.bagging:
            return random.sample(self.memory, batch_size)

        return None

    def __len__(self):
        return len(self.memory)


class Agent:

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    multiple: bool = False
    action_count: int
    size: int
    action_pool: Any
    memory_capacity: int
    rewarder: Callable
    step: int = 0

    def __init__(self, size, action_pool, rewarder, memory_capacity):

        self.size, self.action_pool, self.memory_capacity = size, action_pool, memory_capacity
        self.rewarder = rewarder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.multiple = isinstance(action_pool, (list, np.ndarray))
        self.action_count = action_pool if not self.multiple else np.sum(action_pool)

        self.policy = DQN(size, self.action_count).to(self.device)
        self.target = DQN(size, self.action_count).to(self.device)
        self.target.load_state_dict(self.policy.parameters())

        self.optimizer = RMSprop(self.policy.parameters())
        self.memory = ReplayMemory(memory_capacity)

    def choose_actions(self, state):
        if state is None:
            return None

        sample = random.random()
        eps_threshold = Agent.EPS_END + (Agent.EPS_START - Agent.EPS_END) * np.exp(-1. * self.step / Agent.EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                # TODO(0) action pool
                return self.policy.forward(state).max(1)[1].view(1, 1)
        else:
            if self.multiple:
                actions = np.random.rand(len(self.action_pool)) * self.action_pool
            else:
                actions = [random.randrange(self.action_count)]

            return torch.LongTensor(actions)

    def optimize_model(self, transitions):
        if len(self.memory) < Agent.BATCH_SIZE:
            return

        seen_transitions = self.memory.sample(Agent.BATCH_SIZE)
        if seen_transitions is not None:
            """ Train by replaying past experiences """
            self.optimize_model_batch(seen_transitions)

        self.optimize_model_batch(transitions)
        self.memory.push(transitions)

    def optimize_model_batch(self, transitions):

        states_prev = map(lambda t: t.state, transitions)
        states_curr = map(lambda t: t.state_result, transitions)

        # Compute the actions Q(s_t) and Q'(sc_t)
        outputs_policy = self.policy.forward(states_prev)
        outputs_target = self.target.forward(states_curr)

        # Compute input vector

        # Actions taken previously by the policy net
        actions_batch = torch.tensor(np.vectorize(lambda t: t.action, transitions))
        input = outputs_policy.gather(dim = 1, index = actions_batch.view(-1, 1)).squeeze()

        # Compute target vector

        # Compute the rewards given st, st+1 for t in [0, batch_size]
        rewards = np.vectorize(self.rewarder)(transitions)

        # Compute the expected Q values
        mask = np.array(t.state["alive"] for t in transitions)
        target = rewards + (outputs_target.max(dim = 1).values * Agent.GAMMA) * mask

        # Compute Huber loss
        loss = fc.smooth_l1_loss(input, target)

        # Reset accumulated gradients
        self.optimizer.zero_grad()

        # Perform optimization process
        loss.backward()
        self.optimizer.step()
        self.step += 1

        if self.step % Agent.TARGET_UPDATE == 0:
            self.target.load_state_dict(self.policy.parameters())
