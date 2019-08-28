import pygame
import numpy as np
import os
import torch

from PIL import Image
from typing import Optional
from modules import Track, Sprite, Agent, Transition

TITLE = "RL racer"
SIZE = [700, 700]

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


def create_snapshot(surface, filename: str = None, format = "PNG", save = False, raw = False, tensor = False):
    # Get image data
    data = pygame.surfarray.pixels3d(surface)

    # Preprocess the image
    image = Image.fromarray(np.rollaxis(data, 0, 1)[::-1, :, :], "RGB")
    image = image.rotate(270)

    if raw:
        raw_image = np.asarray(image)
        if tensor:
            raw_image_tensor = torch.FloatTensor(raw_image).transpose(1, 2).transpose(0, 1)

            return raw_image_tensor

        return raw_image

    if save:
        image.save("snapshots/" + filename, format = format)

    # Clear resources
    del data

    return image


def smoothness(x):
    return np.sqrt(np.log10(x))


def rewarder(prev_params: Optional[dict], curr_params: dict):
    if not curr_params["alive"]:
        return -1.0

    reward_acc = smoothness(curr_params["acc"] / curr_params["acc_max"] * (np.e - 1))
    if prev_params is not None and curr_params["acc"] < prev_params["acc"]:
        reward_acc = np.sqrt(reward_acc)

    if curr_params["alive"]:
        reward_pos = 1.0 - smoothness(curr_params["width"] / curr_params["width_max"] * (np.e - 1))
    else:
        reward_pos = 0

    reward = 1.0 + reward_acc + reward_pos

    return reward


def racing_game(agent_active = True, episode_count = 25):

    # Set full screen centered
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    pygame.display.set_caption(TITLE)

    # Set the height and width of the screen
    screen = pygame.display.set_mode(SIZE)
    surface = pygame.display.get_surface()

    # Set the icon
    icon = pygame.image.load("./assets/icon.png")
    icon.set_colorkey(TRANSPARENT)
    pygame.display.set_icon(icon)

    # Loop until the user clicks the close button.
    done = False

    # Create a text renderer helper function
    text_renderer = create_text_renderer(screen)

    # Create the environment
    track = Track()
    track.initialize(SIZE, text_renderer)

    attenuation = 1.0
    if not agent_active:
        # Set up timer for smooth rendering and synchronization
        clock = pygame.time.Clock()
        prev_time = pygame.time.get_ticks()
    else:
        # Set up the agent
        agent = Agent(np.array(SIZE), np.array([Sprite.MOTION_SPACE_COUNT, Sprite.STEERING_SPACE_COUNT]), rewarder)

    # States
    prev_state, curr_state = None, None
    prev_params, curr_params = None, None

    actions = None

    episode_counter = 0

    while not done:

        if not agent_active:
            # Continuous key press
            keys = pygame.key.get_pressed()

            if keys[pygame.K_UP]:
                track.sprite.movement(Sprite.acceleration * attenuation)
            elif keys[pygame.K_DOWN]:
                track.sprite.movement(-Sprite.acceleration * attenuation)

            if keys[pygame.K_LEFT]:
                track.sprite.rotation += track.sprite.steering * attenuation
            elif keys[pygame.K_RIGHT]:
                track.sprite.rotation -= track.sprite.steering * attenuation
        elif prev_state is not None:
            # Generate actions
            actions = agent.choose_actions(prev_state)

            track.sprite.act_actions(actions)

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
                    create_snapshot(surface, filename = "screen.png", save = True)
                elif event.key == pygame.K_r:
                    track.reset_track()

        # Clear the screen and set the screen background
        screen.fill(CLEAR_SCREEN)

        # Environment act
        lap_finished = track.act(attenuation)

        # Render environment
        track.render(screen)

        # Update the screen
        pygame.display.flip()

        if not agent_active:
            # Compute rendering time
            curr_time = pygame.time.get_ticks()
            attenuation, prev_time = (curr_time - prev_time) / (1000 / FPS_CAP), curr_time

            # Handle constant FPS cap
            pygame.display.set_caption("{0}: {1:.2f}".format(TITLE, clock.get_fps()))

            clock.tick(FPS_CAP)
        else:
            curr_state, curr_params = create_snapshot(surface, raw = True, tensor = True), track.get_params()

            # Create a fresh transition
            transition = Transition(prev_state, actions, curr_state, rewarder(prev_params, curr_params))

            # Update the memory state
            agent.memory.push(transition)

            # Optimize model
            agent.optimize_model()

            if lap_finished:
                episode_counter += 1
                if episode_counter == episode_count:
                    done = True
                else:
                    # Initialize the environment and state
                    track.reset_track()

            if not curr_params["alive"]:
                # Initialize the environment and state
                track.reset_track()

            prev_state, curr_state = curr_state, None
            prev_params, curr_params = curr_params, None

    # Be IDLE friendly
    pygame.quit()
