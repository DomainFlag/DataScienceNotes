import numpy as np
import torch
import matplotlib.pyplot as plt

from itertools import count
from modules import Track, Sprite, Transition, Env
from models import DQN, A2C, Base


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
        return -15.0

    # Alive
    reward = 0.

    # Greater motion
    if params["acc"] > 0.0:
        reward_acc = smoothness(params["acc"] / params["acc_max"]) * 3.0
    elif params["acc"] == 0.:
        reward_acc = -1.0
    else:
        reward_acc = -2.0

    # Progress; No penalization for 0 offset as the params are frame dependent
    index_offset = Track.get_index_offset(prev_params["index"], params["index"])
    if index_offset > 0:
        reward += 3.0
    elif index_offset < 0:
        reward -= 3.0
    else:
        reward -= 0.25

    # Keep the line direction
    offset_angle = compute_min_offset(params["angle"], params["rot"], np.pi * 2)
    if not offset_angle == 0 and params["acc"] > 0.0:
        if abs(offset_angle) > 0.35:
            reward_dir = -2.5
        else:
            reward_dir = 1.5 * (1 if offset_angle > 0 else -1)
            if not params["is_to_left"]:
                reward_dir *= -1

            if reward_dir < 0:
                reward_dir = -smoothness(abs(offset_angle) / 0.35) * 0.3
    else:
        reward_dir = 0

    # Reward for reached milestone
    if params["progress_total"] > prev_params["progress_max"]:
        prev_milestone, milestone = prev_params["progress_max"] // 100, params["progress_total"] // 100
        # Case for moving backwards and forward to receive same reward
        if milestone > prev_milestone:
            reward += 5

    # Keep center
    width_offset = min(params["width"], params["width_half"])
    reward += smoothness(width_offset / params["width_half"]) * 3 - 2

    reward += reward_acc + reward_dir

    return reward


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


def racing_game(agent_active = True, agent_live = True, agent_cache = True, agent_interactive = True,
                agent_file = "model_distance.pt", track_cache = True, track_save = False, track_file = "track_model.npy",
                frame_size = (200, 200), episode_count = 700, frame_buffer = False,
                grayscale = True, random_reset = True):
    assert not (not agent_active and agent_live), "Live agent needs to be active"
    assert not (track_cache and track_save), "The track is already cached locally"

    # Environment
    env = Env(frame_size, frame_buffer = frame_buffer, agent_active = agent_active, track_file = track_file,
              track_cache = track_cache, track_save = track_save)

    if not agent_active:
        while not env.done:
            env.step(sync = True)
    else:
        # Set up the agent
        action_space = Sprite.ACTION_SPACE_COUNT * Sprite.MOTION_SPACE_COUNT * Sprite.STEERING_SPACE_COUNT + 1

        size = (1, *frame_size) if grayscale else (3, *frame_size)
        agent: Base = DQN(size, action_space)
        if agent_cache:
            agent.load_network_from_dict(agent_file, agent_live)

            if agent_interactive:
                agent.policy.eval()

                render_reward_(agent.step_reward)

        while not env.done:
            # Render initially
            env.step(sync = False)

            # Init screen
            prev_screen, _, prev_params = env.state(frame_active = True, params_active = True)
            screen = prev_screen
            prev_state = screen - prev_screen
            reward_total = torch.tensor(0.)

            for step in count():

                # Run the action
                action = agent.choose_action(prev_state, agent_live)
                env.track.sprite.act_action(action)

                # Run the env
                env.step(sync = False)

                # Get current state
                screen, _, params = env.state(frame_active = True, params_active = True)
                if params["alive"]:
                    state = screen - prev_screen
                else:
                    state = None

                # Update memory and optimize model
                if not agent_live:
                    reward = torch.tensor(rewarder(prev_params, params)).to(agent.device)
                    reward_total += reward
                    transition = Transition(prev_state, action, state, reward)

                    agent.memory.push(transition)
                    agent.optimize_model()

                # Swap states
                prev_state, prev_screen, prev_params = state, screen, params

                # Reset environment and state when not alive or finished successfully a lap
                if not params["alive"] or params["lap"] > 0:
                    if not agent_live:
                        agent.model_new_episode(params["progress_total"], reward_total.item(), step)
                        env.done = agent.episode >= episode_count

                    env.reset(random_reset=random_reset)
                    break

                if env.done:
                    break

        if not agent_live:
            # Save final model
            agent.save_network_to_dict("model.pt")

    # Release env
    env.release()
