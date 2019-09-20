import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Base
from modules import ReplayMemory
from torch.optim.rmsprop import RMSprop


class Model(nn.Module):

    size: tuple
    actions_count: int

    def __init__(self, size: tuple, actions_count: int):
        super(Model, self).__init__()

        self.size, self.actions_count = size, actions_count
        self.sec1 = nn.Sequential(
            nn.Conv2d(size[0], 8, kernel_size = 2, stride = 1, padding = 0),
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

        features, conv = np.array(size[-2:]), None
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

        # BxCxHxW => BxX => BxA
        return self.ln(x.view(batch_size, -1))


class DQN(Base):
    """ DQN and DDQN agent
    Based on https://arxiv.org/pdf/1509.06461.pdf paper
    """

    BATCH_SIZE = 64
    GAMMA = 0.9999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 3e3
    TARGET_UPDATE = 4e3
    LEARNING_RATE = 4e-3

    def __init__(self, size, action_count, double = True):
        super().__init__(size, action_count)
        self.double = double

        self.policy = Model(size, self.action_count).to(self.device)
        self.target = Model(size, self.action_count).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())

        self.optimizer = RMSprop(self.policy.parameters(), DQN.LEARNING_RATE)
        self.memory = ReplayMemory()

    def choose_action(self, state, agent_live = False):
        eps_threshold = DQN.EPS_END + (DQN.EPS_START - DQN.EPS_END) * np.exp(-1. * self.step / DQN.EPS_DECAY)

        if agent_live or random.random() > eps_threshold:
            with torch.no_grad():
                outputs = self.policy.forward(state.clone().to(self.device).unsqueeze(0))
                action = outputs.squeeze(0).max(0).indices
        else:
            action = torch.tensor(np.random.rand(self.action_count).argmax(), dtype = torch.long).to(self.device)

        return action

    def optimize_model(self):
        if len(self.memory) < DQN.BATCH_SIZE or not self.memory.is_ready():
            return

        # Memory transitions
        transition_batch = self.memory.sample_transition_batch(DQN.BATCH_SIZE)

        state_batch = torch.stack(transition_batch.state).to(self.device, copy = True)
        action_batch = torch.stack(transition_batch.action)
        reward_batch = torch.stack(transition_batch.reward).to(self.device, copy = True)

        # Compute input (outputs_action_policy) - actions q values Q(s_t, a)
        outputs_action_policy = self.policy.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute target (outputs_action_target)
        target = torch.zeros((DQN.BATCH_SIZE,), dtype = torch.float).to(self.device)

        state_mask = torch.tensor([ s is not None for s in transition_batch.next_state ], dtype = torch.bool)
        state_non_final = torch.stack([ s for s in transition_batch.next_state if s is not None ]).to(self.device, copy = True)

        if self.double:
            # More stable version - less likely to pick overestimated values as it's regulated by target network
            policy_output_indices = self.policy.forward(state_non_final).max(1).indices.unsqueeze(1)
            target[state_mask] = self.target.forward(state_non_final).gather(1, policy_output_indices).squeeze(1)
        else:
            target_output = self.target.forward(state_non_final)
            target[state_mask] = target_output.max(1).values

        outputs_action_target = reward_batch + (target * DQN.GAMMA)

        # Compute Huber loss
        loss = F.smooth_l1_loss(outputs_action_target, outputs_action_policy)

        # Reset accumulated gradients and compute gradients
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)

        # Adjust weights
        self.optimizer.step()

        self.step += 1
        self.episode_step += 1

        if self.episode_step % DQN.episode_step_every == 0:
            print(f"\tStep: {self.episode_step}\tReward: {self.memory.reward_acc}")

        if self.step % DQN.TARGET_UPDATE == 0:
            print(f"Step: {int(self.step // DQN.TARGET_UPDATE)}\tUpdating the target net...")
            self.target.load_state_dict(self.policy.state_dict())

    def load_network_from_dict(self, filename, agent_live, verbose = True):
        """ Load a network from the dict """
        checkpoint = super().load_network_from_dict(filename, agent_live, verbose)

        # load network state
        self.policy.load_state_dict(checkpoint["state_model"])
        self.target.load_state_dict(checkpoint["state_model"])
        self.optimizer.load_state_dict(checkpoint["state_optimizer"])

    def save_network_to_dict(self, filename, verbose = False):
        """ Save a network to the dict """

        # save network state
        checkpoint = super().save_network_to_dict(filename, verbose)
        checkpoint.update({
            'state_model': self.policy.state_dict(),
            'state_optimizer': self.optimizer.state_dict()
        })

        # save the network's model as the checkpoint
        torch.save(checkpoint, "./static/" + filename)
