import pygame
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from modules import Track, Sprite, Agent, Transition, State

TITLE = "RL racer"
SIZE = [500, 800]

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
            raw_image_tensor = torch.FloatTensor(raw_image)
            if not grayscale or len(image.getbands()) > 1:
                raw_image_tensor = raw_image_tensor.transpose(1, 2).transpose(0, 1)
            else:
                raw_image_tensor = raw_image_tensor.unsqueeze(dim = 0)

            if normalize:
                raw_image_tensor /= 255.

            return raw_image_tensor, image

        return raw_image, image

    return image


def smoothness(x):
    """ X value is expected to be normalized - [0, 1] """
    assert 0.0 <= x <= 1.0, "x < 0 or x > 1.0, x - %s" % (x,)

    return (np.log(x + 1.0) / np.log(2)) ** (1 / 2)


def compute_min_offset(a, b, cycle):
    offset = a - b

    if abs(offset) < cycle - abs(offset):
        return offset
    else:
        return cycle * (-1 if a > b else 1) + offset


def rewarder(prev_params: dict, params: dict) -> float:
    if not params["alive"]:
        return -30.0

    reward = 0.

    # # Greater motion
    if params["acc"] > 0.0:
        reward_acc = smoothness(params["acc"] / params["acc_max"]) * 2.0
        if params["acc"] <= prev_params["acc"]:
            reward_acc = reward_acc ** 2
    elif params["acc"] == 0.:
        reward_acc = -1.0
    else:
        reward_acc = -2.0

    # # Progress; No penalization for 0 offset as the params are frame dependent
    index_offset = Track.get_index_offset(prev_params["index"], params["index"])
    if index_offset > 0:
        reward += 2.0
    elif index_offset < 0:
        reward -= 3.0
    else:
        reward -= 0.3

    # Keep the line direction
    offset_angle = compute_min_offset(params["angle"], params["rot"], np.pi * 2)
    if not offset_angle == 0:
        if abs(offset_angle) > 0.35:
            reward -= -4.0
        else:
            reward = 3.0 * (1 if offset_angle > 0 else -1)
            if not params["is_to_left"]:
                reward *= -1

    # # Reward for reached milestone
    if params["progress_total"] > prev_params["progress_max"]:
        prev_milestone, milestone = prev_params["progress_max"] // 100, params["progress_total"] // 100
        # Case for moving backwards and forward to receive same reward
        if milestone > prev_milestone:
            reward += 10

    # # Keep center
    width_offset = min(params["width"], params["width_half"])
    reward += smoothness(width_offset / params["width_half"]) * 3 - 2.25

    # # Acc reward + being alive
    reward += reward_acc + 0.5

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


def render_reward_(rewards, step = 5):
    reward_steps = np.array(rewards[:(len(rewards) // step) * step]).reshape(-1, step)

    smooth_path = reward_steps.reshape(-1, step).mean(axis = 1)
    path_deviation = 1.5 + reward_steps.reshape(-1, step).std(axis = 1)
    indices = np.arange(0, len(path_deviation)) * step

    plt.plot(indices, smooth_path, linewidth = 2)
    plt.fill_between(indices, (smooth_path - path_deviation / 2), (smooth_path + path_deviation / 2), color = 'b',
                     alpha = .05)
    plt.title('Reward')
    plt.show()


def racing_game(agent_active = True, agent_live = False, agent_cache = False, agent_interactive = False,
                agent_file = "model_reward.pt", track_cache = True, track_save = False, track_file = "track_model.npy",
                frame_size = (200, 200), frame_discontinuity = 1, episode_count = 750, frame_buffer = True,
                state_seq_size = 5, state_seq_discontinuity = 3, grayscale = True, random_reset = True):
    assert not (not agent_active and agent_live), "Live agent needs to be active"
    assert not (track_cache and track_save), "The track is already cached locally"
    assert (state_seq_size > state_seq_discontinuity), "The seq validation can't be larger than input seq"

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
        action_space = Sprite.ACTION_SPACE_COUNT * Sprite.MOTION_SPACE_COUNT * Sprite.STEERING_SPACE_COUNT + 1

        channels = 1 if grayscale else 3
        agent = Agent(frame_size, channels, action_space, rewarder)
        if agent_cache:
            agent.load_network_from_dict(filename = agent_file)

            if agent_interactive:
                agent.policy.eval()

                render_reward_(agent.step_reward)

    state_seq, params = State(state_seq_size, state_seq_discontinuity), None
    counter = 0
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
                    track.sprite.steer(Sprite.steering * attenuation)
                elif keys[pygame.K_RIGHT]:
                    track.sprite.steer(-Sprite.steering * attenuation)

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

        # Environment act and render
        track.act(attenuation)
        track.render(screen)

        if not frame_buffer:
            # Update the screen
            pygame.display.flip()

        params = track.get_params()
        if not agent_active:
            # Compute rendering time
            curr_time = pygame.time.get_ticks()
            attenuation, prev_time = (curr_time - prev_time) / (1000 / FPS_CAP), curr_time

            # Handle constant FPS cap
            caption_params = [ TITLE, clock.get_fps(), params["index"], params["progress"][1], params["lap"] ]

            clock.tick(FPS_CAP)
        else:
            # Generate env states and params
            state, img = create_snapshot(surface, size = frame_size, center = track.sprite.get_position(), raw = True,
                                         tensor = True, normalize = True, grayscale = True)
            params = track.get_params()
            caption_params = [ TITLE, params["index"], params["progress"][1], params["lap"], 0 ]

            flag = state_seq.push(state, params)
            if flag == 1:
                # Generate actions
                actions = agent.choose_actions(torch.stack(state_seq.state_seq_prev), agent_live = agent_live)
                track.sprite.act_actions(actions)

                state_seq.register(actions)
                counter += 1

            # Create transition and update memory state
            if not agent_live and (flag == 2 or not params["alive"]):
                if params["alive"]:
                    reward = torch.FloatTensor([rewarder(state_seq.prev_params, state_seq.params)]).to(agent.device)
                    transition = Transition(torch.stack(state_seq.state_seq_prev), state_seq.actions,
                                            torch.stack(state_seq.state_seq_prev + state_seq.state_seq), reward)
                else:
                    reward = torch.FloatTensor([rewarder(None, params)]).to(agent.device)

                    # The reward is always negative because it's dead
                    last_full_state = agent.memory.construct_full_state(state_seq.state_seq)
                    transition = Transition(last_full_state, state_seq.actions, None, reward)

                # Update memory and optimize
                agent.memory.push(transition)
                agent.optimize_model()

            if flag == 2:
                state_seq.swap()

            # Reset environment and state when not alive or finished successfully a lap
            if not params["alive"] or params["lap"] > 0:
                if not agent_live:
                    agent.model_new_episode(params["progress_total"], counter)
                    if agent.episode >= episode_count:
                        done = True

                track.reset_track(random_reset = random_reset)
                state_seq.reset()
                counter = 0

        caption_renderer(caption_params)

    # Save final model
    agent.save_network_to_dict("model.pt")

    # Be IDLE friendly
    pygame.quit()
