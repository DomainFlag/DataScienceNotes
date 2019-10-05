import gym
import numpy as np
import torch
import torchvision.transforms as transforms

from modules.envs import BaseEnv
from PIL import Image


class Baseline(BaseEnv):

    ENV_ACTION_SPACE = 2

    # This is based on the code from gym.
    screen_width = 600

    # Environment
    env = None

    def __init__(self, device, frame_size, frame_diff):
        super().__init__(device, frame_diff)

        self.frame_size = frame_size

    def init(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.resizer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size, interpolation = Image.CUBIC),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def get_cart_location(self):
        world_width = self.env.x_threshold * 2
        scale = Baseline.screen_width / world_width

        return int(self.env.state[0] * scale + Baseline.screen_width / 2.0)

    def get_frame(self):
        # transpose into torch order (CHW)
        screen = self.env.render(mode = 'rgb_array').transpose((2, 0, 1))

        # Strip off the top and bottom of the screen
        screen = screen[:, 160:320]
        view_width = 320
        cart_location = self.get_cart_location()
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (Baseline.screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]

        # Convert to float, rescale, convert to torch tensor
        screen = np.ascontiguousarray(screen, dtype = np.float32) / 255
        screen = torch.from_numpy(screen)

        # Resize, and add a batch dimension (BCHW)
        frame = self.resizer(screen)

        return self.edit_frame(frame)

    def step(self, action = None, sync = False):
        reward, params = None, None
        if action is not None:
            _, reward, done, _ = self.env.step(action.item())

            params = {
                'alive': not done
            }

            reward = torch.tensor(reward).to(self.device)

        return self.get_frame(), reward, params

    def reset(self, episode):
        super().reset(episode)

        self.env.reset()

    def release(self):
        self.env.close()
