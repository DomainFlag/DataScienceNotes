import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import io

from torch.optim.rmsprop import RMSprop
from collections import namedtuple
from typing import List, Callable, Optional, Any

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    size: np.ndarray
    actions_count: int

    def __init__(self, size: np.ndarray, channels: int, actions_count: int):
        super(DQN, self).__init__()

        self.size, self.actions_count = size, actions_count
        self.sec1 = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size = 2, stride = 1, padding = 0),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.sec2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.sec3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        features, conv = size.copy(), None
        self.pipeline = [ self.sec1, self.sec2, self.sec3 ]
        for sec in self.pipeline:
            iter = sec.modules()

            _, conv = next(iter), next(iter)

            features = (features + np.array(conv.padding) * 2 - (np.array(conv.kernel_size) - 1) - 1) \
                // np.array(conv.stride) + 1

        input_size = int(features[0] * features[1] * conv.out_channels)

        self.ln = nn.Linear(input_size, actions_count)

    def forward(self, x):
        batch_size = x.size(0)

        for sec in self.pipeline:
            x = sec.forward(x)

        # BxCxHxW => BxA
        return self.ln(x.view(batch_size, -1))


class ReplayMemory:

    MEMORY_CAPACITY: int = 10000
    REPLAY_START: int = 2250

    capacity: int
    memory: List[Transition]
    position: int
    reward_acc: float = 0.

    def __init__(self, capacity: int = None):
        self.memory = []
        self.position = 0
        self.capacity = capacity if capacity is not None else ReplayMemory.MEMORY_CAPACITY

    def push(self, transition: Transition):
        """ Save transition by possibly overriding pre-existing transitions """
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity
        self.reward_acc += getattr(transition, "reward").item()

    def sample_transition_batch(self, batch_size) -> Optional[Transition]:
        transitions = random.sample(self.memory, batch_size)

        return Transition(*zip(*transitions))

    def reset_history(self):
        self.reward_acc = 0.

    def is_ready(self):
        return self.REPLAY_START <= len(self.memory)

    def __len__(self):
        return len(self.memory)


class Agent:

    BATCH_SIZE = 64
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 2e4
    TARGET_UPDATE = 4e3

    learning_rate = 4e-3
    multiple: bool = False
    action_count: int
    action_cum: Optional[torch.Tensor]

    size: int
    action_pool: Any
    rewarder: Callable

    episode: int = 0
    episode_step: int = 0
    episode_step_every: int = 50

    step: int = 0
    step_every: int = 35
    step_reward: list = [0]
    reward_max: float = 0.
    progress_max: float = 0.
    progress_max_step: int = 1

    def __init__(self, size, channels, action_count, rewarder):
        self.size, self.action_count, self.rewarder = size, action_count, rewarder

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Current available device is {self.device.type}")

        self.policy = DQN(size, channels, self.action_count).to(self.device)
        self.target = DQN(size, channels, self.action_count).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())

        self.optimizer = RMSprop(self.policy.parameters())
        self.memory = ReplayMemory()

    def choose_action(self, state, agent_live = False):
        eps_threshold = Agent.EPS_END + (Agent.EPS_START - Agent.EPS_END) * np.exp(-1. * self.step / Agent.EPS_DECAY)

        if agent_live or random.random() > eps_threshold:
            with torch.no_grad():
                outputs = self.policy.forward(state.clone().to(self.device).unsqueeze(0))
                action = outputs.squeeze(0).max(0).indices
        else:
            action = torch.tensor(np.random.rand(self.action_count).argmax(), dtype = torch.long).to(self.device)

        return action

    def optimize_model(self):
        if len(self.memory) < Agent.BATCH_SIZE or not self.memory.is_ready():
            return

        # Memory transitions
        transition_batch = self.memory.sample_transition_batch(Agent.BATCH_SIZE)

        state_batch = torch.stack(transition_batch.state).to(self.device, copy = True)
        action_batch = torch.stack(transition_batch.action)
        reward_batch = torch.stack(transition_batch.reward).to(self.device, copy = True)

        # Compute input (outputs_action_policy) - actions q values Q(s_t, a)
        outputs_action_policy = self.policy.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute target (outputs_action_target)
        target = torch.zeros((Agent.BATCH_SIZE,), dtype = torch.float).to(self.device)

        state_mask = torch.tensor([ s is not None for s in transition_batch.next_state ], dtype = torch.bool)
        state_non_final = torch.stack([ s for s in transition_batch.next_state if s is not None ]).to(self.device, copy = True)

        target_output = self.target.forward(state_non_final)
        target[state_mask] = target_output.max(1).values
        outputs_action_target = reward_batch + (target * Agent.GAMMA)

        # Compute Huber loss
        loss = F.smooth_l1_loss(outputs_action_policy, outputs_action_target)

        # Reset accumulated gradients and compute gradients
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)

        # Adjust weights
        self.optimizer.step()

        self.step += 1
        self.episode_step += 1

        if self.episode_step % Agent.episode_step_every == 0:
            print(f"\tStep: {self.episode_step}\tReward: {self.memory.reward_acc}")

        if self.step % Agent.TARGET_UPDATE == 0:
            print(f"Step: {int(self.step // Agent.TARGET_UPDATE)}\tUpdating the target net...")
            self.target.load_state_dict(self.policy.state_dict())

    def model_new_episode(self, progress, step_count):
        print(f"Episode {self.episode}\tStep: {step_count}\tReward: "
              f"{self.memory.reward_acc:.5f} and progress: {progress}")

        progress_speed = progress / step_count
        if progress > self.progress_max or (progress_speed > self.progress_max / self.progress_max_step and
                                            progress == self.progress_max):
            print(f"\tProgress ({self.progress_max} --> {progress})\tSpeed: {progress_speed:0.3f}.  Saving model ...")

            self.save_network_to_dict("model_distance.pt")
            self.progress_max = progress
            self.progress_max_step = step_count

        if self.memory.reward_acc > self.reward_max:
            print(f"\tReward higher ({self.reward_max:.6f} --> {self.memory.reward_acc:.6f}).  Saving model ...")

            self.save_network_to_dict("model_reward.pt")
            self.reward_max = self.memory.reward_acc

        self.step_reward[-1] = self.memory.reward_acc
        self.step_reward.append(0)
        self.episode += 1
        self.episode_step = 0

        self.memory.reset_history()

    def load_network_from_dict(self, filename):
        """ Load a network from the dict """

        # load network state
        with open("./models/" + filename, 'rb') as file:
            buffer = io.BytesIO(file.read())

        checkpoint = torch.load(buffer, map_location = self.device.type)

        # assign parameters
        self.policy.load_state_dict(checkpoint["state_model"])
        self.target.load_state_dict(checkpoint["state_model"])
        self.optimizer.load_state_dict(checkpoint["state_optimizer"])
        self.episode = checkpoint["episode"]
        self.step_reward = checkpoint["reward_history"]

    def save_network_to_dict(self, filename):
        """ Save a network to the dict """

        # save network state
        checkpoint = {
            'episode': self.episode,
            'reward_history': self.step_reward,
            'state_model': self.policy.state_dict(),
            'state_optimizer': self.optimizer.state_dict()
        }

        # save the network's model as the checkpoint
        torch.save(checkpoint, "./models/" + filename)
