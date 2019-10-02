import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from itertools import count
from modules import Track, Sprite, Baseline, BaseEnv, Env
from models import DQN, A2C, Base
from typing import Optional

RACE_VERSION = 0.816


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


def rewarder_simple(prev_params: dict, params: dict) -> float:
    if not params["alive"]:
        return -1

    # Keep center
    width_offset = min(params["width"], params["width_half"]) / params["width_half"]

    # Incentivize cumulative arithmetic progress | alive
    if not width_offset > 0.3:
        reward = -0.5
    else:
        if prev_params['progress_max'] < params['progress_max']:
            p0, p1 = prev_params['progress_max'] // 25,  params['progress_max'] // 25
            if p0 != p1:
                reward = 1
            else:
                reward = 0
        else:
            reward = -0.25

    return reward


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
    if params["progress_total"][0] > prev_params["progress_max"]:
        prev_milestone, milestone = prev_params["progress_max"] // 250, params["progress_total"][0] // 250
        # Case for moving backwards and forward to receive same reward
        if milestone > prev_milestone:
            reward += 5

    # Keep center
    width_offset = min(params["width"], params["width_half"])
    reward += smoothness(width_offset / params["width_half"]) * 3 - 2

    reward += reward_acc + reward_dir

    return reward


def agent_display_(data, title, step = 20, save = False):
    if len(data) < step:
        return

    data_steps = np.array(data[:(len(data) // step) * step]).reshape(-1, step)

    smooth_path = data_steps.reshape(-1, step).mean(axis = 1)
    path_deviation = 1.5 + data_steps.reshape(-1, step).std(axis = 1)
    indices = np.arange(0, len(path_deviation)) * step

    plt.plot(indices, smooth_path, linewidth = 2)
    plt.fill_between(indices, (smooth_path - path_deviation / 2), (smooth_path + path_deviation / 2), color = 'b',
                     alpha = .05)
    plt.title(title)
    if save:
        plt.savefig(f"./static/{title}_{RACE_VERSION}.png")
    plt.show()


def agent_act(env, agent, locker, episode_count, agent_live, random_reset, reset_every):

    while not env.exit:

        # Render initially
        env.step(action = None, sync = False)

        # Init screen
        state, _, prev_params = env.state(frame_active = True, frame_diff = True, params_active = True)

        for step in count():

            # Run the action
            action, residuals = agent.choose_action(state, agent_live)
            reward, env.done = env.step(action, sync = False, device = agent.device)

            # Get current state
            state, _, params = env.state(frame_active = True, frame_diff = True, params_active = True)

            if env.done:
                state = None

            # Optimize model
            if not agent_live and reward is None:
                reward = torch.tensor(rewarder_simple(prev_params, params)).to(agent.device)

            if not agent_live:
                env.done = agent.optimize_model(env.prev_state, action, state, reward, done = env.done,
                                                residuals = residuals, locker = locker) or env.done

            # Swap states
            prev_params = params

            # Reset environment and state when not alive or finished successfully a lap
            if env.done:
                progress = params["progress_total"][1] if params is not None else step

                agent.model_new_episode(progress, step, agent_live)
                if env.done and not agent_live:
                    # Checking in case of explicit exit
                    env.exit = agent.episode >= episode_count or env.exit

                env.reset(random_reset = random_reset, hard_reset = agent.episode % reset_every == 0)
                break

            # Exit time
            if env.exit:
                break


def racing_game(agent_active = True, agent_live = False, agent_cache = False, agent_interactive = False,
                agent_file = "model.pt", track_cache = True, track_save = False, track_file = "track_model.npy",
                frame_size = (150, 150), episode_count = 300, frame_buffer = False,
                grayscale = True, random_reset = True, multi_threading = True, num_processes = 3, asynchronous = False,
                reset_every = 6):
    assert not (not agent_active and agent_live), "Live agent needs to be active"
    assert not (track_cache and track_save), "The track is already cached locally"

    print(f"Running on {RACE_VERSION} ...")

    agent: Optional[Base] = None
    if agent_active:
        # Set up the agent
        action_space = Baseline.ENV_ACTION_SPACE

        size = (1, *frame_size) if grayscale else (3, *frame_size)
        agent = A2C(size, action_space, recurrent = False)
        if not agent_live and agent_active and multi_threading:
            mp.set_start_method('spawn')

            agent.model.share_memory()

        if agent_cache:
            agent.load_network_from_dict(agent_file, agent_live)

            if agent_interactive:
                agent.eval()

                # Display training info
                agent_display_(agent.reward_history, title = "Reward")
                agent_display_(agent.progress_history, title = "Progress")

    # Initialize environment
    envs = []
    if agent_live and agent_active:
        num_processes = 1
    else:
        assert(num_processes > 0)

    for t in range(num_processes):
        env: BaseEnv = Baseline(frame_size, frame_buffer = frame_buffer, agent_active = agent_active, track_file = track_file,
                  track_cache = track_cache, track_save = track_save)

        if agent is not None:
            env.attach_device(agent.device)

        envs.append(env)

    # Perform agent on environment
    if not agent_active:
        while not envs[0].exit:
            envs[0].step(sync = True)
    else:
        processes = []
        locker = mp.Lock() if not asynchronous else None

        # Act agent on environment
        for env in envs:
            process = mp.Process(target = agent_act, args = (env, agent, locker, episode_count, agent_live, random_reset, reset_every))
            process.start()
            processes.append(process)

        # Wait for the processes to finish
        for process in processes:
            process.join()

        if not agent_live:
            # Save final model
            agent.save_network_to_dict("model.pt", verbose = True)

            # Display training info
            agent_display_(agent.reward_history, title = "Reward", save = True)
            agent_display_(agent.progress_history, title = "Progress", save = True)

    # Release env
    for t in range(num_processes):
        envs[t].release()
