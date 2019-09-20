import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Base
from torch import distributions
from torch.optim.rmsprop import RMSprop


class Model(nn.Module):
    """ Based on https://arxiv.org/pdf/1602.01783.pdf paper"""

    size: np.ndarray
    actions_count: int

    def __init__(self, size: np.ndarray, channels: int, actions_count: int, hidden_size: int = 512, num_layers: int = 2):
        super(Model, self).__init__()

        self.size, self.actions_count = size, actions_count
        self.hidden_size, self.num_layers = hidden_size, num_layers

        # Start of non-output layers to be shared
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
        self.rnn = nn.GRU(input_size = input_size, hidden_size = self.hidden_size, num_layers = self.num_layers,
                           batch_first = True, bidirectional = False)

        # Start of independent output layers
        # Actor - fully connected layer
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU,
            nn.Linear(64, self.actions_count)
        )

        # Critic - fully connected layer
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU,
            nn.Linear(64, 1)
        )

    def forward(self, x, hidden_state):
        batch_size, seq_size = x.size(0), x.size(1)

        # BxSxCxHxW => BSxCxHxW
        x = x.view(-1, *x.shape[-3:])
        for sec in self.pipeline:
            x = sec.forward(x)

        # BSxCxHxW => BxSxI
        x, hidden_state = self.rnn(x.view(batch_size, seq_size, -1), hidden_state)

        # BxSxI => BxI | [-1]
        x = x[:, -1, :]

        # Policy dictates the action taken
        policy = F.softmax(self.actor(x), dim = 1)

        # BxI => Bx1 => B
        value = self.critic(x).squeeze(1)

        return policy, value, hidden_state


class A2C(Base):

    BATCH_SIZE = 64
    GAMMA = 0.9999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 3e3
    TARGET_UPDATE = 4e3
    LEARNING_RATE = 4e-3

    def __init__(self, size, action_count):
        super().__init__(size, action_count)

        self.optimizer = RMSprop(self.policy.parameters(), A2C.LEARNING_RATE)

        pass

    def choose_action(self, state, agent_live = False):
        pass

    def optimize_model(self):
        pass

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
