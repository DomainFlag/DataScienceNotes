import torch
import io
import modules.utils as utils
import time

from itertools import count


class BaseAgent:

    size: tuple
    action_count: int

    step: int = 0

    episode: int = 0
    episode_step: int = 0
    episode_step_every: int = 50

    reward_acc: float = 0.
    reward_history: list = []

    progress_history: list = []

    agent_cache: bool
    agent_cache_name: str

    def __init__(self, device, size, action_count, agent_cache = False, agent_cache_name = 'model.pt',):
        self.device, self.size, self.action_count = device, size, action_count
        self.agent_cache, self.agent_cache_name = agent_cache, agent_cache_name

    def choose_action(self, state, training = False):
        raise not NotImplementedError

    def model_optimize(self, prev_state, action, state, reward, done = False) -> bool:
        raise NotImplementedError

    def model_train(self, envs, episode_count):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def model_new_episode(self, progress, step_count):
        print(f"Episode {self.episode}\tStep: {step_count}\tReward: {self.reward_acc:.5f} and progress: {progress}")

        self.reward_history.append(self.reward_acc)
        self.progress_history.append(progress)
        self.episode += 1
        self.episode_step, self.reward_acc = 0, 0.

    def load_network_from_dict(self, filename, training, verbose = True):
        """ Load a network from the dict """
        if verbose:
            print(f"Loading model from {filename} ...")

        with open("./static/" + filename, 'rb') as file:
            buffer = io.BytesIO(file.read())

        checkpoint = torch.load(buffer, map_location = self.device.type)

        # if not training:
        #     self.episode = checkpoint["episode"]
        #     self.reward_history = checkpoint["reward_history"]
        #     self.progress_history = checkpoint["progress_history"]

        return checkpoint

    def save_network_to_dict(self, filename = None, verbose = False):
        """ Save a network to the dict """
        if filename is None:
            filename = self.agent_cache_name

        if verbose:
            print(f"Saving model to {filename} ...")

        checkpoint = {
            'episode': self.episode,
            'reward_history': self.reward_history,
            'progress_history': self.progress_history
        }

        return checkpoint

    def model_valid(self, env, episode_count):

        # Set model to eval mode
        self.eval()

        # Initialize env
        env.init()

        while not env.exit:

            # Init screen and render initially
            state, _, _ = env.step(action = None)

            for step in count():

                # Run the action and get state
                action, residuals = self.choose_action(state, training = False)
                state, reward, params = env.step(action)

                # Reset environment and state when not alive or finished successfully a lap
                if env.done:
                    progress = params["progress"] if params is not None else step

                    self.model_new_episode(progress, step)
                    if env.done:
                        # Checking in case of explicit exit
                        env.exit = self.episode >= episode_count or env.exit

                    env.reset(self.episode)
                    break

                # Exit time
                if env.exit:
                    break

        # Display valid info
        utils.show_features(self.reward_history, title = "Reward", workers = 1)
        utils.show_features(self.progress_history, title = "Progress", workers = 1)

        # Release env
        env.release()
