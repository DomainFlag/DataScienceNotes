import pygame
import numpy as np
import os
import torch

from PIL import Image
from modules import Track, Sprite

TITLE = "RL racer"
SIZE = [500, 500]

FPS_CAP = 60.0
TRANSPARENT = (0, 0, 0, 0)
CLEAR_SCREEN = (255, 255, 255)


def create_text_renderer(screen):
    # Load font
    font = pygame.font.SysFont('Comic Sans MS', 24)

    def text_render(text, position):
        surface = font.render(text, False, (0, 0, 0))

        # Render to current surface
        screen.blit(surface, position)

    return text_render


def create_snapshot(surface, size = None, center = None, filename: str = "screen.png", format = "PNG", save = False,
                    raw = False, normalize = False, tensor = False, grayscale = False):
    # Get image data
    data = pygame.surfarray.pixels3d(surface)

    # Preprocess the image
    image = Image.fromarray(np.rollaxis(data, 0, 1)[::-1, :, :], mode = "RGB")
    image = image.rotate(270)
    if size is not None and center is not None:
        lu = np.maximum((center - size / 2).astype(int), (0, 0))
        rl = lu + size

        image = image.crop(box = (*lu, *rl))

    if grayscale:
        image = image.convert(mode = "L")

    if save:
        image.save("./snapshots/" + filename, format = format)

    if raw:
        raw_image = np.asarray(image)
        if tensor:
            raw_image_tensor = torch.from_numpy(raw_image)
            if not grayscale or len(image.getbands()) > 1:
                raw_image_tensor = raw_image_tensor.transpose(1, 2).transpose(0, 1)
            else:
                raw_image_tensor = raw_image_tensor.unsqueeze(dim = 0)

            if normalize:
                raw_image_tensor /= 255.

            return raw_image_tensor.float(), image

        return raw_image, image

    return image


def get_caption_renderer(window_active, clock = False):
    if clock:
        message = "{:s}: {:.2f}fps, index - {:d}, progress - {:5.2f}%, lap - {:d}"
    else:
        message = "{:s}: index - {:d}, progress - {:5.2f}%, lap - {:d}, episode - {:d}"

    def renderer(args):
        if window_active:
            pygame.display.set_caption(message.format(*args))

    return renderer


class Env:

    attenuation: float = 1.0
    frame_buffer: bool
    agent_active: bool

    def __init__(self, frame_size, frame_buffer = False, agent_active = False, track_file = None, track_cache = True,
                 track_save = False):
        self.frame_size = frame_size
        self.frame_buffer = frame_buffer
        self.agent_active = agent_active

        # Set full screen centered and hint audio for dsp instead of als
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        os.environ['SDL_AUDIODRIVER'] = 'dsp'

        # Initialize Pygame modules
        pygame.init()

        if not frame_buffer:
            # Set the height and width of the screen
            self.surface = pygame.display.set_mode(SIZE)

            # Set the icon
            icon = pygame.image.load("./assets/icon.png")
            icon.set_colorkey(TRANSPARENT)

            pygame.display.set_icon(icon)
        else:
            self.surface = pygame.Surface(SIZE)

        # Window caption renderer
        caption_renderer = get_caption_renderer(not frame_buffer, clock=not agent_active)

        # Loop until the user clicks the close button.
        self.done = False

        # Create a text renderer helper function
        text_renderer = create_text_renderer(self.surface)

        # Create the environment
        self.track = Track()
        self.track.initialize_track(SIZE, text_renderer, track_save=track_save, track_cache=track_cache,
                               filename = track_file)
        self.track.initialize_sprite()
        if not agent_active:
            # Set up timer for smooth rendering and synchronization
            self.clock = pygame.time.Clock()
            self.prev_time = pygame.time.get_ticks()

    def state(self, frame_active = False, params_active = False):
        """ Generate env states and params """
        frame, img, params = None, None, None

        if frame_active:
            sprite_pos = self.track.sprite.get_position()
            frame, img = create_snapshot(self.surface, size=self.frame_size, center=sprite_pos, raw=True,
                                         tensor=True, grayscale=True)

        if params_active:
            params = self.track.get_params()

        return frame, img, params

    def step(self, sync = False):
        # Handle key events
        self.event_handler()

        # Clear the screen and set the screen background
        self.surface.fill(CLEAR_SCREEN)

        # Environment act and render
        self.track.act(self.attenuation)
        self.track.render(self.surface)

        if not self.frame_buffer:
            # Update the screen
            pygame.display.flip()

        if sync:
            # Compute rendering time
            self.curr_time = pygame.time.get_ticks()
            self.attenuation, self.prev_time = (self.curr_time - self.prev_time) / (1000 / FPS_CAP), self.curr_time

            # Handle constant FPS cap
            self.clock.tick(FPS_CAP)

    def event_handler(self):
        # Event queue while window is active
        if not self.frame_buffer:

            # if not self.agent_active:
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
                    self.done = True

                # Escape key is pressed
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.done = True
                    elif event.key == pygame.K_PRINT:
                        create_snapshot(self.surface, filename="screen.png", save=True)
                    elif event.key == pygame.K_r:
                        self.track.reset_track()

    def reset(self, random_reset = False):
        self.track.reset_track(random_reset = random_reset)

    def release(self):
        # Be IDLE friendly
        pygame.quit()