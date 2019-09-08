import pygame
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from modules import Track, Sprite, Agent, Transition

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
                    raw = False, normalize = False, tensor = False):
    # Get image data
    data = pygame.surfarray.pixels3d(surface)

    # Preprocess the image
    image = Image.fromarray(np.rollaxis(data, 0, 1)[::-1, :, :], "RGB")
    image = image.rotate(270)
    if size is not None and center is not None:
        lu = np.maximum((center - size / 2), (0, 0))
        rl = lu + size

        image = image.crop(box = (*lu, *rl))

    if save:
        image.save("./snapshots/" + filename, format = format)

    if raw:
        raw_image = np.asarray(image)

        if tensor:
            raw_image_tensor = torch.FloatTensor(raw_image).transpose(1, 2).transpose(0, 1)

            if normalize:
                raw_image_tensor /= 255.

            return raw_image_tensor, image

        return raw_image, image

    return image


def smoothness(x):
    """ X value is expected to be normalized - [0, 1] """
    assert 0.0 <= x <= 1.0, "x < 0 or x > 1.0, x - %s" % (x,)

    return np.log(x + 1.0) / np.log(2)


def rewarder(prev_params: dict, curr_params: dict):
    if not curr_params["alive"]:
        return -1.0

    reward = 0.

    # Greater motion
    if curr_params["acc"] >= 0.0:
        reward_acc = smoothness(curr_params["acc"] / curr_params["acc_max"]) * 0.25
        if curr_params["acc"] <= prev_params["acc"]:
            reward_acc = reward_acc ** 2
    else:
        reward_acc = -2.0

    # Progress
    if Track.get_index_offset(prev_params["index"], curr_params["index"]) > 0:
        reward += 0.75
    else:
        reward -= 2.0

    # Direction
    offset_abs_angle = abs(curr_params["angle"] - curr_params["rot"])
    offset_angle = min(offset_abs_angle, abs(offset_abs_angle - np.pi * 2))
    if offset_angle >= 0.3:
        reward -= 0.75
    else:
        reward += 0.5

    # Keep center
    width_offset = min(curr_params["width"], curr_params["width_half"])
    track_center_offset = 1.0 - width_offset / curr_params["width_half"]
    reward_pos = -0.5 if track_center_offset > 0.4 else 0.5

    # Acc reward + being alive
    reward += 0.2 + reward_acc + reward_pos

    return reward


def get_caption_renderer(window_active, clock = False):
    if clock:
        message = "{:s}: {:.2f}fps, index - {:d}, progress - {:5.2f}%, lap - {:d}"
    else:
        message = "{:s}: index - {:d}, progress - {:5.2f}%, lap - {:d}, episode - {:d}"

    def renderer(args):
        if window_active:
            pygame.display.set_caption(message.format(*args))

    return renderer


def render_reward_(rewards):
    plt.plot(np.arange(0, len(rewards)), rewards)
    plt.title('Reward')
    plt.show()


def racing_game(agent_active = True, agent_live = False, agent_cache = False, agent_interactive = False,
                agent_file = "model_reward.pt", track_cache = True, track_save = False, track_file = "track_model.npy",
                frame_size = (200, 200), episode_count = 300, frame_buffer = True):
    assert not (not agent_active and agent_live), "Live agent needs to be active"
    assert not (track_cache and track_save), "The track is already cached locally"

    # Set full screen centered and hint audio for dsp instead of als
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    # Setting default frame size
    frame_size = np.array(frame_size) if frame_size is not None else np.array(SIZE)

    # Initialize pygame modules
    pygame.init()

    if not frame_buffer:
        # Set the height and width of the screen
        screen = pygame.display.set_mode(SIZE)
        surface = pygame.display.get_surface()

        # Set the icon
        icon = pygame.image.load("./assets/icon.png")
        icon.set_colorkey(TRANSPARENT)
        pygame.display.set_icon(icon)
    else:
        screen = pygame.Surface(SIZE)
        surface = screen

    # Window caption renderer
    caption_renderer = get_caption_renderer(not frame_buffer, clock = not agent_active)

    # Loop until the user clicks the close button.
    done = False

    # Create a text renderer helper function
    text_renderer = create_text_renderer(screen)

    # Create the environment
    track = Track()
    track.initialize_track(SIZE, text_renderer, track_save = track_save, track_cache = track_cache,
                           filename = track_file)
    track.initialize_sprite()

    attenuation = 1.0
    if not agent_active:
        # Set up timer for smooth rendering and synchronization
        clock = pygame.time.Clock()
        prev_time = pygame.time.get_ticks()
    else:
        # Set up the agent
        action_space = Sprite.ACTION_SPACE_COUNT + Sprite.MOTION_SPACE_COUNT * Sprite.STEERING_SPACE_COUNT + 1

        agent = Agent(frame_size, action_space, rewarder)
        if agent_cache:
            agent.load_network_from_dict(filename = agent_file)

            if agent_interactive:
                render_reward_(agent.step_reward)

    state, params = None, None
    while not done:

        # Event queue while window is active
        if not frame_buffer:
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
        track.act(attenuation)

        # Render environment
        track.render(screen)

        if not frame_buffer:
            # Update the screen
            pygame.display.flip()

        if not agent_active:
            # Compute rendering time
            curr_time = pygame.time.get_ticks()
            attenuation, prev_time = (curr_time - prev_time) / (1000 / FPS_CAP), curr_time

            # Handle constant FPS cap
            params_next = track.get_params()
            caption_params = [ TITLE, clock.get_fps(), params_next["index"], params_next["progress"][1],
                               params_next["lap"] ]

            clock.tick(FPS_CAP)
        else:
            state_next, state_img = create_snapshot(surface, size = frame_size, center = track.sprite.get_position(), raw = True,
                                         tensor = True, normalize = True, save = False)
            params_next = track.get_params(state = state_img, centered = True)

            # Generate actions
            actions = agent.choose_actions(state)
            track.sprite.act_actions(actions)

            if not agent_live:
                # Create transition and update memory state
                if state is not None:
                    reward = torch.FloatTensor([rewarder(params, params_next)]).to(agent.device)
                    actions_aligned = agent.reshape_actions(actions).to(agent.device)

                    transition = Transition(state, actions_aligned, state_next, reward)
                    agent.memory.push(transition)

                # Optimize model
                agent.optimize_model()

            # Caption parameters
            caption_params = [ TITLE, params_next["index"], params_next["progress"][1], params_next["lap"],
                               agent.episode ]

            if not params_next["alive"]:
                # Initialize the environment and state
                track.reset_track()
                agent.model_new_episode()

                state_next, params_next = None, None

                if agent.episode >= episode_count:
                    done = True
            else:
                # Caption parameters
                caption_params = [ TITLE, params_next["index"], params_next["progress"][1], params_next["lap"],
                                   -1 ]

            state, params = state_next, params_next

        caption_renderer(caption_params)

    # Be IDLE friendly
    pygame.quit()
