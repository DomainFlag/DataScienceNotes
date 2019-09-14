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
    batch_size: int
    channels: int

    def __init__(self, size: np.ndarray, channels: int, actions_count: int, hidden_size = 768, num_layers = 2):
        super(DQN, self).__init__()

        self.size, self.actions_count = size, actions_count
        self.hidden_size, self.num_layers = hidden_size, num_layers

        self.sec1 = nn.Sequential(
            nn.Conv2d(channels, 6, kernel_size = 2, stride = 1, padding = 0),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )

        self.sec2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

        self.sec3 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(16),
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
        self.ln = nn.Linear(self.hidden_size, actions_count)

    def forward(self, inputs):
        batch_size, seq_size = inputs.size(0), inputs.size(1)

        # BxSxCxHxW => BSxCxHxW
        x = inputs.view(-1, *inputs.shape[-3:])
        for sec in self.pipeline:
            x = sec.forward(x)

        # BSxCxHxW => BxSxI
        x, _ = self.rnn(x.view(batch_size, seq_size, -1))

        # Fully connected layer and BxSxI => BxI | S[-1]
        return self.ln(x)[:, -1, :]


class ReplayMemory:

    MEMORY_CAPACITY: int = 3250
    REPLAY_START: int = 325

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

    def construct_full_state(self, curr_state):
        last_state = self.sample_recent_transition()
        state = getattr(last_state, "state")
        prev_state = getattr(last_state, "next_state")

        if len(curr_state) == 0:
            full_state = torch.cat((state, prev_state))
        else:
            full_state = torch.cat((state, prev_state, torch.stack(curr_state)))

        return full_state[-state.size(0):]

    def sample_recent_transition(self):
        return self.memory[(self.position - 1) % self.capacity]

    def reset_history(self):
        self.reward_acc = 0.

    def is_ready(self):
        return self.REPLAY_START <= len(self.memory)

    def __len__(self):
        return len(self.memory)


class State:

    prev_params: Optional[dict] = None
    params: Optional[dict] = None
    state_seq_prev: List = []
    state_seq: List = []
    actions = None

    def __init__(self, seq_size: int, seq_discontinuity: int):
        self.seq_size = seq_size
        self.seq_discontinuity = seq_discontinuity

    def push(self, state, params) -> int:
        if len(self.state_seq_prev) < self.seq_size:
            self.state_seq_prev.append(state)
            self.prev_params = params

            if len(self.state_seq_prev) == self.seq_size:
                return 1
        elif len(self.state_seq) < self.seq_discontinuity:
            self.state_seq.append(state)
            self.params = params
        else:
            return 2

        return 0

    def register(self, actions):
        self.actions = actions

    def swap(self):
        self.prev_params, self.params = self.params, None
        self.state_seq_prev, self.state_seq = self.state_seq.copy(), []

    def reset(self):
        self.prev_params, self.params = None, None
        self.state_seq_prev, self.state_seq = [], []
        self.actions = None


class Agent:

    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 430
    TARGET_UPDATE = 7e2

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
    step_loss: list = [0]
    step_reward: list = [0]
    step_loss_min: float = np.inf
    reward_max: float = 0.
    progress_max: float = 0.
    progress_max_step: int = 1

    def __init__(self, size, channels, action_pool, rewarder):

        self.size, self.action_pool, self.rewarder = size, action_pool, rewarder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Current available device is {self.device.type}")

        self.multiple = isinstance(action_pool, (list, np.ndarray))
        self.action_count = action_pool if not self.multiple else np.sum(action_pool)
        self.action_cum = (torch.cumsum(torch.from_numpy(self.action_pool), dim = 0) - self.action_pool[0]).to(self.device) \
            if self.multiple else None

        self.policy = DQN(size, channels, self.action_count).to(self.device)
        self.target = DQN(size, channels, self.action_count).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())

        self.optimizer = RMSprop(self.policy.parameters())
        self.memory = ReplayMemory()

    def choose_actions(self, state_seq, agent_live = False):
        if state_seq is None:
            return None

        sample = random.random()
        eps_threshold = Agent.EPS_END + (Agent.EPS_START - Agent.EPS_END) * np.exp(-1. * self.step / Agent.EPS_DECAY)

        if sample > eps_threshold or agent_live:
            with torch.no_grad():
                outputs = self.policy.forward(state_seq.clone().to(self.device).unsqueeze(0)).squeeze(0)
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

    def optimize_model(self, loss_tracking = False):
        if len(self.memory) < Agent.BATCH_SIZE or not self.memory.is_ready():
            return

        # Memory transitions
        transition_batch = self.memory.sample_transition_batch(Agent.BATCH_SIZE)

        state_batch = torch.stack(transition_batch.state).to(self.device, copy = True)
        action_batch = torch.stack(transition_batch.action)
        reward_batch = torch.cat(transition_batch.reward)

        # Compute input (outputs_action_policy) - actions q values Q(s_t, a)
        outputs_action_policy = self.policy.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute target (outputs_action_target)
        target = torch.zeros((Agent.BATCH_SIZE,), dtype = torch.float).to(self.device)

        state_mask = torch.tensor([ s is not None for s in transition_batch.next_state ], dtype = torch.bool)
        state_non_final = torch.stack([ s for s in transition_batch.next_state if s is not None ]).to(self.device, copy = True)

        target[state_mask] = self.unpack_actions(self.target.forward(state_non_final))
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
        self.step_loss[-1] += loss.item()

        if self.episode_step % Agent.episode_step_every == 0:
            print(f"\tStep: {self.episode_step}\tReward: {self.memory.reward_acc}")

        if loss_tracking and self.step % Agent.step_every == 0:
            step_loss_cur = self.step_loss[-1] / self.step_every
            self.step_loss.append(0)

            print(f"Step: {self.step // Agent.step_every}\tTraining Loss: {step_loss_cur:.6f}")
            if step_loss_cur < self.step_loss_min:
                print(f"Loss decreased ({self.step_loss_min:.6f} --> {step_loss_cur:.6f}).  Saving model ...")

                self.save_network_to_dict("model.pt")
                self.step_loss_min = step_loss_cur

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
        self.step_loss = checkpoint["loss_history"]
        self.step_reward = checkpoint["reward_history"]

    def save_network_to_dict(self, filename):
        """ Save a network to the dict """

        # save network state
        checkpoint = {
            'episode': self.episode,
            'loss_history': self.step_loss,
            'loss_model': self.step_loss[-1] / Agent.step_every,
            'reward_history': self.step_reward,
            'state_model': self.policy.state_dict(),
            'state_optimizer': self.optimizer.state_dict()
        }

        # save the network's model as the checkpoint
        torch.save(checkpoint, "./models/" + filename)
