import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Base
from torch import distributions
from torch.optim.rmsprop import RMSprop


class Model(nn.Module):
    """ Based on https://arxiv.org/pdf/1602.01783.pdf paper"""

    size: tuple
    actions_count: int

    def __init__(self, size: tuple, actions_count: int, recurrent = False, hidden_size: int = 512, num_layers: int = 1):
        super(Model, self).__init__()

        self.size, self.actions_count = size, actions_count
        self.recurrent = recurrent
        self.hidden_size, self.num_layers = hidden_size, num_layers

        # Start of non-output layers to be shared
        self.sec1 = nn.Sequential(
            nn.Conv2d(size[0], 8, kernel_size = 5, stride = 2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.sec2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size = 5, stride = 2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.sec3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 5, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # self.sec4 = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size = 5, stride = 2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )

        features, conv = np.array(size[-2:]), None
        self.pipeline = [ self.sec1, self.sec2, self.sec3 ]
        for sec in self.pipeline:
            iter = sec.modules()

            _, conv = next(iter), next(iter)

            features = (features + np.array(conv.padding) * 2 - (np.array(conv.kernel_size) - 1) - 1) \
                // np.array(conv.stride) + 1

        input_size = int(features[0] * features[1] * conv.out_channels)
        if recurrent:
            self.rnn = nn.GRU(input_size = input_size, hidden_size = self.hidden_size, num_layers = self.num_layers,
                              batch_first = True, bidirectional = False)

            input_size = self.hidden_size

        # Start of independent output layers
        # Actor - fully connected layer
        self.actor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.actions_count)
        )

        # Critic - fully connected layer
        self.critic = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, hidden_state):
        assert(len(x.shape) == 5)
        batch_size, seq_size = x.size(0), x.size(1)

        # BxSxCxHxW => BSxCxHxW
        x = x.view(-1, *x.shape[-3:])
        for sec in self.pipeline:
            x = sec.forward(x)

        if self.recurrent:
            # BSxCxHxW => BxSxI
            x, hidden_state = self.rnn(x.view(batch_size, seq_size, -1), hidden_state)

            # BxSxI => BxI | [-1]
            x = x[:, -1, :]
        else:
            # BSxCxHxW => BxI | S = 1
            x = x.view(batch_size, -1)

        # Policy dictates the action to be taken
        policy = F.softmax(self.actor(x), dim = 1)

        # BxI => Bx1 => B
        value = self.critic(x).squeeze(1)

        return policy, value, hidden_state


class A2C(Base):

    GAMMA = 0.95
    TAU = 1.0
    ENT_START = 0.005
    ENT_END = 0.0001
    ENT_DECAY = 2e2
    LEARNING_RATE = 3e-5
    VALUE_COEF = 0.05
    T_STEP = 20
    STEP_MAX = 300

    rewards: list = []
    values: list = []
    probs_log: list = []
    entropy: torch.tensor
    hidden_state = None
    recurrent: bool

    def __init__(self, size, action_count, recurrent = True):
        super().__init__(size, action_count)

        self.recurrent = recurrent
        self.entropy = torch.tensor(0., device = self.device)
        self.model = Model(self.size, self.action_count, recurrent = self.recurrent).to(self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr = A2C.LEARNING_RATE)

    def choose_action(self, state, agent_live = False):
        if agent_live:
            with torch.no_grad():
                policy, value, self.hidden_state = self.model(state.unsqueeze(0).unsqueeze(0), self.hidden_state)
        else:
            policy, value, self.hidden_state = self.model(state.unsqueeze(0).unsqueeze(0), self.hidden_state)

        action = policy.multinomial(1).data[0, 0]

        return action, (policy, value)

    def optimize_model(self, prev_state, action, state, reward, done = False, residuals = None, locker = None):

        # Unpacking residuals variables
        policy, value = residuals

        prob_log = torch.log(policy[0])
        self.probs_log.append(prob_log[action])
        self.entropy += -torch.sum(policy[0] * prob_log).to(self.device)

        self.values.append(value[0])
        self.rewards.append(reward)

        self.step += 1
        self.episode_step += 1
        self.reward_acc += reward.item()

        if self.episode_step % A2C.episode_step_every == 0:
            print(f"\tStep: {self.episode_step}\tReward: {self.reward_acc}")

        if done or self.episode_step % A2C.T_STEP == 0 or self.episode_step == A2C.STEP_MAX:
            # Discard accumulated gradients
            self.optimizer.zero_grad()

            rewards = torch.zeros((len(self.rewards),)).to(self.device)
            estimations = torch.zeros((len(self.values,))).to(self.device)
            if not done:
                # Finished existing state
                rewards[-1] = self.values[-1]
                estimations[-1] = self.values[-1]

            for t in reversed(range(len(self.rewards) - 1)):
                rewards[t] = rewards[t + 1] * A2C.GAMMA + self.rewards[t]

                # Generalized advantage estimation
                estimations[t] = self.rewards[t] + A2C.GAMMA * self.values[t + 1] - self.values[t]
                estimations[t] = estimations[t + 1] * A2C.GAMMA * A2C.TAU + estimations[t]

            # Entropy strength
            self.entropy *= (A2C.ENT_END + (A2C.ENT_START - A2C.ENT_END) * np.exp(-1. * self.step / A2C.ENT_DECAY))

            # Advantage: how good an action is
            advantages = rewards - torch.stack(self.values)

            # MSE error
            value_loss = A2C.VALUE_COEF * advantages.pow(2).mean()
            policy_loss = (torch.stack(self.probs_log) * estimations).mean()

            # Gradient ascent on policy, entropy - reward and exploration | descent on value - minimize critic penalty
            loss = value_loss - policy_loss - self.entropy.to(self.device)

            # Compute gradients
            loss.backward()

            # Clip the weights
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)

            if locker is not None:
                locker.acquire()

            print("STARTED")

            # Adjust weights
            self.optimizer.step()

            print("ENDED")
            if locker is not None:
                locker.release()

            self.probs_log, self.values, self.rewards = [], [], []
            self.entropy = torch.tensor(0., device = self.device)
            if self.recurrent:
                # Don't backward past graph
                self.hidden_state = self.hidden_state.detach()

        if self.episode_step == A2C.STEP_MAX:
            return True

        return False

    def eval(self):
        self.model.eval()

    def model_new_episode(self, progress, step_count, agent_live):
        super().model_new_episode(progress, step_count, agent_live)

        self.hidden_state = None
        self.entropy = torch.tensor(0., device = self.device)
        self.probs_log, self.values, self.rewards = [], [], []

    def load_network_from_dict(self, filename, agent_live, verbose = True):
        """ Load a network from the dict """
        checkpoint = super().load_network_from_dict(filename, agent_live, verbose)

        # load network state
        self.model.load_state_dict(checkpoint["state_model"])
        self.optimizer.load_state_dict(checkpoint["state_optimizer"])

    def save_network_to_dict(self, filename, verbose = False):
        """ Save a network to the dict """

        # save network state
        checkpoint = super().save_network_to_dict(filename, verbose)
        checkpoint.update({
            'state_model': self.model.state_dict(),
            'state_optimizer': self.optimizer.state_dict()
        })

        # save the network's model as the checkpoint
        torch.save(checkpoint, "./static/" + filename)