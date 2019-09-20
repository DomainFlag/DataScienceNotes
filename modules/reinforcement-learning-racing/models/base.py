import torch
import torch.nn as nn
import io


class Base(nn.Module):

    size: tuple
    action_count: int

    global_step: int = 0

    episode: int = 0
    episode_step: int = 0
    episode_step_every: int = 50

    reward_history: list = []
    reward_max: float = 0.

    progress_history: list = []
    progress_max: float = 0.
    progress_max_step: int = 1

    def __init__(self, size, action_count):
        super(Base, self).__init__()
        self.size, self.action_count = size, action_count

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Current available device is {self.device.type}")

    def choose_action(self, state, agent_live = False):
        raise not NotImplementedError

    def optimize_model(self):
        raise NotImplementedError

    def model_new_episode(self, progress, reward, step_count):
        print(f"Episode {self.episode}\tStep: {step_count}\tReward: "
              f"{reward:.5f} and progress: {progress}")

        progress_speed = progress / step_count
        if progress > self.progress_max or (progress_speed > self.progress_max / self.progress_max_step and
                                            progress == self.progress_max):
            print(f"\tProgress ({self.progress_max} --> {progress})\tSpeed: {progress_speed:0.3f}.  Saving model ...")

            self.save_network_to_dict("model_distance.pt")
            self.progress_max = progress
            self.progress_max_step = step_count

        if reward > self.reward_max:
            print(f"\tReward higher ({self.reward_max:.6f} --> {reward:.6f}).  Saving model ...")

            self.save_network_to_dict("model_reward.pt")
            self.reward_max = reward

        self.reward_history.append(reward)
        self.progress_history.append(progress)
        self.episode += 1
        self.episode_step = 0

    def load_network_from_dict(self, filename, agent_live, verbose = True):
        """ Load a network from the dict """
        if verbose:
            print(f"Loading model from {filename} ...")

        with open("./static/" + filename, 'rb') as file:
            buffer = io.BytesIO(file.read())

        checkpoint = torch.load(buffer, map_location = self.device.type)

        self.reward_max = checkpoint["reward_max"]
        self.progress_max = checkpoint["progress_max"]

        if agent_live:
            self.episode = checkpoint["episode"]
            self.reward_history = checkpoint["reward_history"]
            self.progress_history = checkpoint["progress_history"]

        return checkpoint

    def save_network_to_dict(self, filename, verbose = False):
        """ Save a network to the dict """
        if verbose:
            print(f"Saving model to {filename} ...")

        checkpoint = {
            'episode': self.episode,
            'reward_history': self.reward_history,
            'progress_history': self.progress_history,
            'reward_max': self.reward_max,
            'progress_max': self.progress_max
        }

        return checkpoint
