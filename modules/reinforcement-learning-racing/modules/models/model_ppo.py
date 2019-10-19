import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import modules.utils as utils

from itertools import count
from heapq import merge
from modules.models import BaseAgent
from torch.optim.adam import Adam

from collections import namedtuple

Sample = namedtuple('Sample', ('states', 'lives', 'prob_log', 'actions', 'rewards'))


class Model(nn.Module):

    size: tuple
    actions_count: int

    def __init__(self, size: tuple, actions_count: int, recurrent = False, hidden_size: int = 1024, num_layers: int = 2):
        super(Model, self).__init__()

        self.size, self.actions_count = size, actions_count
        self.recurrent = recurrent
        self.hidden_size, self.num_layers = hidden_size, num_layers

        # Start of non-output layers to be shared
        self.sec1 = nn.Sequential(
            nn.Conv2d(size[0], 32, kernel_size = 3, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.sec2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = 3, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.sec3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = 3, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        features, conv = np.array(size[-2:]), None
        self.pipeline = [self.sec1, self.sec2, self.sec3]
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
            nn.Tanh(),
            nn.Linear(64, self.actions_count)
        )

        # Critic - fully connected layer
        self.critic = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, hidden_state):
        assert (len(x.shape) == 5)
        batch_size, seq_size = x.size(0), x.size(1)

        # BxSxCxHxW => BSxCxHxW
        x = x.view(-1, *x.shape[-3:])
        for sec in self.pipeline:
            x = sec.forward(x)

        # BSxCxHxW => BxSxI
        if self.recurrent:
            x, hidden_state = self.rnn(x.view(batch_size, seq_size, -1), hidden_state)
        else:
            x = x.view(batch_size, seq_size, -1)

        # Policy dictates the action to be taken | BxSxP
        policy = F.softmax(self.actor(x), dim = 2)

        # BxSxI => BxSx1 => BxS
        value = self.critic(x).squeeze(2)

        return policy, value, hidden_state


class PPO(BaseAgent):
    """ PPO agent
    Based on https://arxiv.org/pdf/1707.06347.pdf paper
    """
    AGENT_NAME = 'PPO'

    GAMMA = 0.99
    TAU = 0.95
    ENT_START = 0.01
    ENT_END = 0.001
    ENT_DECAY = 2e2
    LEARNING_RATE = 3e-4
    VALUE_COEF = 0.5
    CLIP_ALPHA = 0.2
    EPOCHS_COUNT = 6
    T_STEP = 750
    STEP_MAX = 2250

    states: list = []
    lives: list = []
    actions: list = []
    rewards: list = []
    probs_log: list = []
    progress: list = []

    prev_hidden_state = None
    hidden_state = None

    recurrent: bool
    asynchronous: bool

    def __init__(self, device, size, action_count, agent_cache = False, agent_cache_name = 'model.pt',
                 recurrent = False,
                 num_processes = 4):
        super().__init__(device, size, action_count, agent_cache, agent_cache_name)

        self.recurrent, self.num_processes = recurrent, num_processes

        self.policy = Model(self.size, self.action_count, recurrent = self.recurrent).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr = PPO.LEARNING_RATE)

        self.target = Model(self.size, self.action_count, recurrent = self.recurrent).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())

        # Load network
        if self.agent_cache:
            self.load_network_from_dict(self.agent_cache_name, False)

    def choose_action(self, state, model = None, training = False, batch_seq_ready = False):
        assert (model is not None), 'A model must be picked for inference'

        if not batch_seq_ready:
            state = state.unsqueeze(0).unsqueeze(0)

        if not training:
            with torch.no_grad():
                policy, value, self.hidden_state = model(state, self.hidden_state)
        else:
            policy, value, self.hidden_state = model(state, self.hidden_state)

        if not batch_seq_ready:
            actions = policy.multinomial(1).data[0, 0]
        else:
            # BxSxP => BxSx1 | 1 action => BxS
            actions = policy.multinomial(1).squeeze()

        return actions, (policy, value)

    def eval(self):
        self.policy.eval()

    def model_optimize(self, prev_state, action, state, reward, done = False, residuals = None, queue = None, cond = None):

        # Unpacking residuals variables
        policy, value = residuals

        if done:
            reward = torch.tensor(0).to(self.device)

        self.probs_log.append(torch.log(policy[0][0][action]))
        self.states.append(prev_state)
        self.actions.append(action)
        self.lives.append(done)
        self.rewards.append(reward)

        self.step += 1
        self.episode_step += 1
        self.reward_acc += reward.item()

        if self.episode_step % PPO.episode_step_every == 0:
            print(f"\tStep: {self.episode_step}\tReward: {self.reward_acc}")

        if self.step % PPO.T_STEP == 0:
            # Discounted rewards
            rewards = torch.zeros((len(self.rewards),)).to(self.device)
            rewards[-1] = self.rewards[-1]
            for t in reversed(range(len(self.rewards) - 1)):
                if self.lives[t]:
                    # Finished existing state
                    rewards[t] = self.rewards[t]
                else:
                    rewards[t] = rewards[t + 1] * PPO.GAMMA + self.rewards[t]

            # Rewards normalization
            rewards = (rewards - rewards.mean()) / rewards.std()

            # Training sample
            sample = Sample(
                torch.stack(self.states),
                self.lives,
                torch.stack(self.probs_log),
                torch.stack(self.actions),
                rewards
            )

            queue.put(sample, block = True)

            with cond:
                # Wait the target model to be updated with the policy
                cond.wait()

            self.states, self.lives, self.actions = [], [], []
            self.probs_log, self.rewards = [], []

            # Don't backward past graph
            if self.recurrent:
                if self.hidden_state is not None:
                    self.hidden_state = self.hidden_state.detach()

                self.prev_hidden_state = self.hidden_state

            return 2

        if self.episode_step == PPO.STEP_MAX:
            return 1

        return 0

    def model_optimize_(self, batch_samples):

        # Unpacking the batch samples
        batch_prob_log = torch.stack(batch_samples.prob_log).to(self.device)
        batch_states = torch.stack(batch_samples.states).to(self.device)
        batch_actions = torch.stack(batch_samples.actions).to(self.device)
        batch_rewards = torch.stack(batch_samples.rewards).to(self.device)

        # Update episode
        self.episode += 1

        # Entropy strength
        entropy_strength = (PPO.ENT_END + (PPO.ENT_START - PPO.ENT_END) * np.exp(-1. * self.episode / PPO.ENT_DECAY))

        # Optimize policy for K epochs:
        for _ in range(PPO.EPOCHS_COUNT):

            # Reset hidden state
            if self.recurrent:
                if self.prev_hidden_state is not None:
                    self.hidden_state = self.prev_hidden_state.clone()
                else:
                    self.hidden_state = None

            # B x S x P policy estimations and B x S values
            _, (policies, values) = self.choose_action(batch_states, model = self.policy, training = True, batch_seq_ready = True)
            policy_probs_log = torch.log(policies)

            # B x S x P | Action Indices (B x S => B x S x 1) => B x S of action prob
            policy_prob_log = policy_probs_log.gather(dim = 2, index = batch_actions.unsqueeze(2)).squeeze(-1)

            # Compute Entropy | B x S x P => B x S => B
            entropy = -torch.sum(policies * policy_probs_log, dim = 2).mean(dim = 1).to(self.device) * entropy_strength

            # Advantage: how good the actions were
            advantages = batch_rewards - values

            # Batch MSE error
            value_loss = PPO.VALUE_COEF * advantages.pow(2).mean(dim = 1)

            # Compute the surrogate objective loss
            ratio = torch.exp(policy_prob_log - batch_prob_log)
            objective_loss = ratio * advantages
            objective_loss_clipped = torch.clamp(ratio, 1 - PPO.CLIP_ALPHA, 1 + PPO.CLIP_ALPHA) * advantages

            # Compute batch policy loss
            policy_loss = torch.min(objective_loss, objective_loss_clipped).mean(dim = 1)

            # Gradient ascent on policy, entropy - reward and exploration | descent on value - critic penalty
            loss = (value_loss - policy_loss - entropy).mean()

            # Discard accumulated gradients
            self.optimizer.zero_grad()

            # Compute gradients
            loss.backward()

            # Normalize gradients
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

            # Adjust weights
            self.optimizer.step()

        # Update target with new policy states
        self.target.load_state_dict(self.policy.state_dict())

    def model_new_episode(self, progress, step_count):
        self.progress.append((time.time(), progress, self.reward_acc))

        super().model_new_episode(progress, step_count)

        self.hidden_state = None

    def load_network_from_dict(self, filename, agent_live, verbose = True):
        """ Load a network from the dict """
        checkpoint = super().load_network_from_dict(filename, agent_live, verbose)

        # load network state
        self.policy.load_state_dict(checkpoint["state_model"])
        self.target.load_state_dict(checkpoint["state_model"])
        self.optimizer.load_state_dict(checkpoint["state_optimizer"])

    def save_network_to_dict(self, filename = None, verbose = False):
        """ Save a network to the dict """
        if filename is None:
            filename = self.agent_cache_name

        # save network state
        checkpoint = super().save_network_to_dict(filename, verbose)
        checkpoint.update({
            'state_model': self.policy.state_dict(),
            'state_optimizer': self.optimizer.state_dict()
        })

        # save the network's model as the checkpoint
        torch.save(checkpoint, "./static/" + filename)

    def model_train(self, envs, episode_count):
        mp.set_start_method('spawn', force = True)

        # Share the agent model memory
        self.target.share_memory()

        # Perform agent on environment
        processes, results = mp.Pool(len(envs)), []

        # Initialize a lock for progress synchronization
        locker = mp.Manager().Lock()
        queue = mp.Manager().Queue()
        cond = mp.Manager().Condition(locker)

        # Act agent on environment
        for env in envs:
            results.append(processes.apply_async(target, (env, self, episode_count, queue, cond)))

        processes.close()

        for _ in range(episode_count):
            samples = []
            for i in range(len(envs)):
                # Wait until all training samples are available
                samples.append(queue.get(block = True))

            # Stop training when max episodes is reached once
            batch_samples = Sample(*zip(*samples))

            # Model batch optimization
            self.model_optimize_(batch_samples)

            with cond:
                # Notify all working processes
                cond.notify_all()

        # Sync workers with the main process
        processes.join()

        # Agent progress; unpacking only progress and reward
        progress = merge(*[res.get() for res in results])
        self.progress_history, self.reward_history = list(zip(*progress))[1:]

        # Display training info
        utils.show_features(self.reward_history, title = "Reward", workers = self.num_processes, save = True)
        utils.show_features(self.progress_history, title = "Progress", workers = self.num_processes, save = True)

        # Save final model
        self.save_network_to_dict(self.agent_cache_name, verbose = True)


# Training function
def target(env, agent, episode_count, queue, cond):
    # Initialize env
    env.init()

    # Training episodes counter
    episodes = 0

    while not env.exit:

        # Init screen and render initially
        state, _, _ = env.step(action = None)

        for step in count():

            # Run the action and get state
            action, residuals = agent.choose_action(state, model = agent.target, training = False)
            state, reward, params = env.step(action)

            if not params['alive']:
                state = None

            flag = agent.model_optimize(env.prev_state, action, state, reward, done = not params['alive'],
                                        residuals = residuals, queue = queue, cond = cond)

            if flag > 0 or not params['alive']:
                env.done = True

            if flag == 2:
                episodes += 1

                # Maximum episodes
                env.exit = episodes == episode_count or env.exit

            # Reset environment and state when not alive
            if env.done:
                progress = params["progress"] if "progress" in params else step
                agent.model_new_episode(progress, step)

                env.reset(agent.episode)
                break

            # Exit time
            if env.exit:
                break

    # Release env
    env.release()

    return agent.progress
