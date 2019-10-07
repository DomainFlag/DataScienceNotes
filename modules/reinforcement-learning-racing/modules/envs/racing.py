import pygame
import numpy as np
import os
import torch
import modules.utils as utils

from PIL import Image
from modules.envs import BaseEnv
from modules import Track, Sprite

TITLE = "RL racer"
SIZE = [700, 700]

FPS_CAP = 60.0
TRANSPARENT = (0, 0, 0, 0)
CLEAR_SCREEN = (255, 255, 255)


def get_caption_renderer(window_active, clock = False):
    if clock:
        message = "{:s}: {:.2f}fps, index - {:d}, progress - {:5.2f}%, lap - {:d}"
    else:
        message = "{:s}: index - {:d}, progress - {:5.2f}%, lap - {:d}, episode - {:d}"

    def renderer(args):
        if window_active:
            pygame.display.set_caption(message.format(*args))

    return renderer


def create_text_renderer(screen):
    # Load font
    font = pygame.font.SysFont('Comic Sans MS', 24)

    def text_render(text, position):
        surface = font.render(text, False, (0, 0, 0))

        # Render to current surface
        screen.blit(surface, position)

    return text_render


def create_snapshot(surface, size = None, center = None, offset = None, rotation = None, filename: str = "screen.png",
                    format = "PNG",
                    save = False,
                    normalize = False, tensor = False, grayscale = False):
    # Get image data
    data = pygame.surfarray.pixels3d(surface)

    # Preprocess the image
    image = Image.fromarray(np.rollaxis(data, 0, 1)[::-1, :, :], mode = "RGB")
    image = image.rotate(270)
    if center is not None and rotation is not None:
        image = image.rotate(rotation, center = tuple(center.astype(int)))

    if size is not None and center is not None:
        if offset is not None:
            center += offset

        lu = np.maximum((center - size / 2).astype(int), (0, 0))
        rl = lu + size

        image = image.crop(box = (*lu, *rl))

    if grayscale:
        image = image.convert(mode = "L")

    if save:
        image.save("./snapshots/" + filename, format = format)

    raw_image = np.asarray(image)
    if tensor:
        raw_image_tensor = torch.from_numpy(raw_image).float()
        if not grayscale or len(image.getbands()) > 1:
            raw_image_tensor = raw_image_tensor.transpose(1, 2).transpose(0, 1)
        else:
            raw_image_tensor = raw_image_tensor.unsqueeze(dim = 0)

        if normalize:
            raw_image_tensor = raw_image_tensor / 255.

        return raw_image_tensor

    return raw_image


def rewarder(params: dict) -> float:
    if not params["alive"]:
        return -1

    reward = 0

    # Keep center
    width_offset = min(params["width"], params["width_half"]) / params["width_half"]
    if not width_offset > 0.4:
        reward -= 2e-1

    # Keep the line direction
    offset_angle = utils.compute_min_offset(params["angle"], params["rot"], np.pi * 2)
    if abs(offset_angle) > 0.3:
        reward -= 2e-1

    if reward == 0:
        reward = 1

    return float(reward)


def rewarder(params: dict) -> float:
    if not params["alive"]:
        return -20

    reward = -1e-1

    if params["progress_max_prev"] < params["progress_max"]:
        reward += params["progress_max"] - params["progress_max_prev"]

    return float(reward)


class Racing(BaseEnv):

    ENV_ACTION_SPACE = 5

    attenuation: float = 1.0

    agent_active: bool
    frame_buffer: bool

    track_random_reset: bool
    track_random_reset_every: int

    def __init__(self, device, frame_size, agent_active = True, track_random_reset = False,
                 track_random_reset_every = 6, frame_diff = False,
                 frame_buffer = False, track_cache = True, track_file = None, track_save = False):
        super().__init__(device, frame_diff)

        self.frame_size = np.array(frame_size)
        self.agent_active, self.frame_buffer = agent_active, frame_buffer
        self.track_random_reset, self.track_random_reset_every = track_random_reset, track_random_reset_every
        self.track_cache, self.track_file, self.track_save = track_cache, track_file, track_save

    def init(self):

        # Set full screen centered and hint audio for dsp instead of als
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        os.environ['SDL_AUDIODRIVER'] = 'dsp'

        # Initialize Pygame modules
        pygame.init()

        if not self.frame_buffer:
            # Set the height and width of the screen
            self.surface = pygame.display.set_mode(SIZE)

            # Set the icon
            icon = pygame.image.load("./assets/icon.png")
            icon.set_colorkey(TRANSPARENT)

            pygame.display.set_icon(icon)
        else:
            self.surface = pygame.Surface(SIZE)

        # Window caption renderer
        caption_renderer = get_caption_renderer(not self.frame_buffer, clock = not self.agent_active)

        # Create a text renderer helper function
        text_renderer = create_text_renderer(self.surface)

        # Create the environment
        self.track = Track()
        self.track.initialize_track(SIZE, text_renderer, track_save = self.track_save, track_cache = self.track_cache,
                                    filename = self.track_file)
        self.track.initialize_sprite()
        if not self.agent_active:
            # Set up timer for smooth rendering and synchronization
            self.clock = pygame.time.Clock()
            self.prev_time = pygame.time.get_ticks()

    def step(self, action = None, sync = False):
        """ Generate env states and params based on action """

        # Handle key events
        self.event_handler()

        # Clear the screen and set the screen background
        self.surface.fill(CLEAR_SCREEN)

        # Environment act and render
        if action is not None:
            self.track.sprite.act_action(action)

        self.track.act(self.attenuation)
        self.track.render(self.surface)

        if not self.frame_buffer:
            # Update the screen
            pygame.display.flip()

        if not self.agent_active and sync:
            # Compute rendering time
            self.curr_time = pygame.time.get_ticks()
            self.attenuation, self.prev_time = (self.curr_time - self.prev_time) / (1000 / FPS_CAP), self.curr_time

            # Handle constant FPS cap
            self.clock.tick(FPS_CAP)

        # Create an image frame
        rot = -(self.track.sprite.rotation / np.pi * 180)
        center = self.track.sprite.get_position()
        offset = np.array([self.frame_size[0] / 2 - 25, 0])

        frame = create_snapshot(self.surface, size = self.frame_size, center = center, offset = offset, tensor = True,
                                rotation = rot, grayscale = True, normalize = True)
        frame = self.edit_frame(frame)

        # Get current env params a reward
        params = self.track.get_params()
        reward = torch.tensor(rewarder(params)).to(self.device)

        # Check if it's done or not
        self.done = self.done or not params["alive"]

        return frame, reward, params

    def event_handler(self):
        # Event queue while window is active
        if not self.frame_buffer:

            if not self.agent_active:
                # Continuous key press
                keys = pygame.key.get_pressed()

                if keys[pygame.K_UP]:
                    self.track.sprite.movement(Sprite.acceleration * self.attenuation)
                elif keys[pygame.K_DOWN]:
                    self.track.sprite.movement(-Sprite.acceleration * self.attenuation)

                if keys[pygame.K_LEFT]:
                    self.track.sprite.steer(Sprite.steering * self.attenuation)
                elif keys[pygame.K_RIGHT]:
                    self.track.sprite.steer(-Sprite.steering * self.attenuation)

            # User did something
            for event in pygame.event.get():
                # Close button is clicked
                if event.type == pygame.QUIT:
                    self.exit = True

                # Escape key is pressed
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit = True
                    elif event.key == pygame.K_PRINT:
                        create_snapshot(self.surface, filename = "screen.png", save = True)
                    elif event.key == pygame.K_r and not self.agent_active:
                        self.track.reset_track()

    def reset(self, episode):
        super().reset(episode)

        self.track.reset_track(random_reset = self.track_random_reset,
                               hard_reset = episode % self.track_random_reset_every == 0)

    def release(self):
        # Be IDLE friendly
        pygame.quit()
