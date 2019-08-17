import pygame
import numpy as np
import os

from PIL import Image
from modules import Track, Sprite


TITLE = "RL racer"
SIZE = [700, 700]

FPS_CAP = 60.0
CLEAR_SCREEN = (255, 255, 255)


def create_text_renderer(screen):
    # Load font
    font = pygame.font.SysFont('Comic Sans MS', 24)

    def text_render(text, position):
        surface = font.render(text, False, (0, 0, 0))

        # Render to current surface
        screen.blit(surface, position)

    return text_render


def create_snapshot(filename: str, surface, format = "PNG"):
    data = pygame.surfarray.pixels3d(surface)
    image = Image.fromarray(np.rollaxis(data, 0, 1)[::-1, :, :], "RGB")
    image = image.rotate(270)
    image.save("snapshots/" + filename, format = format)
    del data


def smoothness(x):
    return np.sqrt(np.log10(x))


def rewarder(prev_params, curr_params):
    reward = 1.0 if curr_params["alive"] else -1.0

    reward_acc = smoothness(curr_params["acc"] / curr_params["acc_max"] * (np.e - 1))
    if curr_params["acc"] < prev_params["acc"]:
        reward_acc = np.sqrt(reward_acc)

    if curr_params["alive"]:
        reward_pos = 1.0 - smoothness(curr_params["width"] / curr_params["width_max"] * (np.e - 1))
    else:
        reward_pos = 0

    reward += reward_acc + reward_pos

    return reward


def racing_game():
    # Set full screen centered
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    pygame.display.set_caption(TITLE)

    # Set the height and width of the screen
    screen = pygame.display.set_mode(SIZE)
    surface = pygame.display.get_surface()

    # Loop until the user clicks the close button.
    done = False

    # Create a text renderer helper function
    text_renderer = create_text_renderer(screen)

    # Create the environment
    track = Track()
    track.initialize(SIZE, text_renderer)

    start_pos, start_rot = track.get_metadata(index = 0)

    sprite = Sprite(np.array(start_pos), start_rot, index = 0)
    sprite.initialize()

    # Set up timer for smooth rendering and synchronization
    clock = pygame.time.Clock()
    prev_time, attenuation = pygame.time.get_ticks(), 0
    while not done:

        # Continuous key press
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            sprite.movement(Sprite.acceleration * attenuation)
        elif keys[pygame.K_DOWN]:
            sprite.movement(-Sprite.acceleration * attenuation)

        if keys[pygame.K_LEFT]:
            sprite.rotation += sprite.steering * attenuation
        elif keys[pygame.K_RIGHT]:
            sprite.rotation -= sprite.steering * attenuation

        # User did something
        for event in pygame.event.get():
            # Close button is clicked
            if event.type == pygame.QUIT:
                done = True

            # Escape key is pressed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                elif event.key == pygame.K_PRINT:
                    create_snapshot("screen.png", surface)

        # Clear the screen and set the screen background
        screen.fill(CLEAR_SCREEN)

        # Render environment
        track.render(screen)
        sprite.render(screen, track, track.track_offset, hint = True)

        # Update the screen
        pygame.display.flip()

        # Compute rendering time
        curr_time = pygame.time.get_ticks()
        attenuation, prev_time = (curr_time - prev_time) / (1000 / FPS_CAP), curr_time

        # Sprite act
        sprite.act(attenuation)

        # Handle constant FPS cap
        pygame.display.set_caption("{0}: {1:.2f}".format(TITLE, clock.get_fps()))

        clock.tick(FPS_CAP)

    # Be IDLE friendly
    pygame.quit()
