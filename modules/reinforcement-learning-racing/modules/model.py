import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as fc

from torch.optim.rmsprop import RMSprop
from collections import namedtuple
from typing import List, Callable, Optional, Any

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    size: np.ndarray
    actions_count: int

    def __init__(self, size: np.ndarray, actions_count: int):
        super(DQN, self).__init__()

        self.size, self.actions_count = size, actions_count

        self.sec1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.sec2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.sec3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = 5, stride = 2, padding = 2),
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

        self.ln = nn.Linear(features[0] * features[1] * conv.out_channels, actions_count)

    def forward(self, inputs):
        for sec in self.pipeline:
            inputs = sec.forward(inputs)

        return self.ln(inputs.view(inputs.size(0), -1))


class ReplayMemory:

    MEMORY_CAPACITY: int = 3000

    bagging: float = 0.34
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

    def sample_batch_transition(self, batch_size, bagging = False) -> Optional[List[Transition]]:
        if not bagging or np.random.random() < ReplayMemory.bagging:
            return random.sample(self.memory, batch_size)

        return None

    def sample_transition_batch(self, batch_size, bagging = False) -> Optional[Transition]:
        transitions = self.sample_batch_transition(batch_size, bagging = bagging)

        if transitions is not None:
            return Transition(*zip(*transitions))

        return None

    def reset_history(self):
        self.reward_acc = 0.

    def __len__(self):
        return len(self.memory)


class Agent:

    BATCH_SIZE = 64
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 500
    TARGET_UPDATE = 10

    multiple: bool = False
    action_count: int
    action_cum: Optional[torch.Tensor]

    size: int
    action_pool: Any
    rewarder: Callable
    episode: int = 0
    step: int = 0
    step_every: int = 50
    step_loss: list = [0]
    step_reward: list = [0]
    step_loss_cur: float = 0.
    step_loss_min: float = np.inf
    reward_max: float = 0.

    def __init__(self, size, action_pool, rewarder):

        self.size, self.action_pool, self.rewarder = size, action_pool, rewarder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.multiple = isinstance(action_pool, (list, np.ndarray))
        self.action_count = action_pool if not self.multiple else np.sum(action_pool)
        self.action_cum = (torch.cumsum(torch.from_numpy(self.action_pool), dim = 0) - self.action_pool[0]).to(self.device) \
            if self.multiple else None

        self.policy = DQN(size, self.action_count).to(self.device)
        self.target = DQN(size, self.action_count).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())

        self.optimizer = RMSprop(self.policy.parameters())
        self.memory = ReplayMemory()

    def choose_actions(self, state):
        if state is None:
            return None

        sample = random.random()
        eps_threshold = Agent.EPS_END + (Agent.EPS_START - Agent.EPS_END) * np.exp(-1. * self.step / Agent.EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                outputs = self.policy.forward(state.clone().to(self.device).unsqueeze(0)).squeeze(0)
                if self.multiple:
                    actions = outputs.view(2, 3).max(1).indices
                else:
                    actions = outputs.max(0).indices

                return actions
        else:
            if self.multiple:
                prob = np.random.rand(self.action_count)
                acts = []

                acc = 0
                for step in self.action_pool:
                    acts.append(prob[acc:acc + step].argmax())
                    acc += step

                actions = np.array(acts)
            else:
                actions = np.random.rand(self.action_count).argmax()

            if self.multiple:
                return torch.LongTensor(actions).to(self.device)
            else:
                return torch.tensor(actions, dtype = torch.long).to(self.device)

    def reshape_actions(self, actions):
        if self.action_cum is not None:
            return actions + self.action_cum

        return actions

    def unpack_actions(self, actions, tensor = True):
        if self.multiple:
            values = []
            batch_actions = actions.detach().cpu().numpy()
            for action in batch_actions:
                values.append([ action[index_start:index_start + self.action_pool[index]].max()
                                for index, index_start in enumerate(self.action_cum) ])

            actions = np.array(values)
        else:
            actions = actions.detach().max(1).values

        if tensor and not isinstance(actions, torch.Tensor):
            return torch.from_numpy(actions).to(self.device)

        return actions

    def optimize_model(self):
        if len(self.memory) < Agent.BATCH_SIZE:
            return

        # Memory transitions
        transition_batch = self.memory.sample_transition_batch(Agent.BATCH_SIZE)

        state_batch = torch.stack(transition_batch.state).clone().to(self.device)
        action_batch = torch.stack(transition_batch.action)
        reward_batch = torch.cat(transition_batch.reward)

        # Compute input (outputs_action_policy) - actions q values Q(s_t, a)
        outputs_action_policy = self.policy.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute target (outputs_action_target)
        target = torch.zeros((Agent.BATCH_SIZE,), dtype = torch.float).to(self.device)

        state_mask = torch.tensor([ s is not None for s in transition_batch.next_state ], dtype = torch.uint8)
        state_non_final = torch.stack([ s for s in transition_batch.next_state if s is not None ]).clone().to(self.device)

        target[state_mask] = self.unpack_actions(self.target.forward(state_non_final))
        outputs_action_target = reward_batch + (target * Agent.GAMMA)

        # Compute Huber loss
        loss = fc.smooth_l1_loss(outputs_action_policy, outputs_action_target)

        # Reset accumulated gradients and compute gradients
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)

        # Adjust weights
        self.optimizer.step()
        self.step += 1
        self.step_loss[-1] += loss.item()

        if self.step % Agent.step_every == 0:
            self.step_loss_cur = self.step_loss[-1] / self.step_every
            self.step_loss.append(0)

            print(f"Step: {self.step // Agent.step_every}\tTraining Loss: {self.step_loss_cur:.6f}")
            if self.step_loss_cur < self.step_loss_min:
                print(f"Loss decreased ({self.step_loss_min:.6f} --> {self.step_loss_cur:.6f}).  Saving model ...")

                self.save_network_to_dict("model.pt")
                self.step_loss_min = self.step_loss_cur

        if self.step % Agent.TARGET_UPDATE == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def model_new_episode(self):
        print(f"\tEpisode {self.episode}\tReward: {self.memory.reward_acc:.5f}")
        if self.memory.reward_acc > self.reward_max:
            print(f"\tReward higher ({self.reward_max:.6f} --> {self.memory.reward_acc:.6f}).  Saving model ...")

            self.save_network_to_dict("model_reward.pt")
            self.reward_max = self.memory.reward_acc

        self.step_reward[-1] = self.memory.reward_acc
        self.step_reward.append(0)
        self.episode += 1

        self.memory.reset_history()

    def load_network_from_dict(self, filename):
        """ Load a network from the dict """

        # load network state
        checkpoint = torch.load("./models/" + filename, map_location = self.device.type)

        # assign parameters
        self.policy.load_state_dict(checkpoint["state_model"])
        self.target.load_state_dict(checkpoint["state_model"])
        self.optimizer.load_state_dict(checkpoint["state_optimizer"])
        self.episode = checkpoint["episode"]
        self.step_loss = checkpoint["loss_history"]

    def save_network_to_dict(self, filename):
        """ Save a network to the dict """

        # save network state
        checkpoint = {
            'episode': self.episode,
            'loss_history': self.step_loss,
            'loss_model': self.step_loss[-1] / Agent.step_every,
            'reward_history': self.step_reward,
            'state_model': self.target.state_dict(),
            'state_optimizer': self.optimizer.state_dict()
        }

        # save the network's model as the checkpoint
        torch.save(checkpoint, "./models/" + filename)
