import torch

from modules.envs import Baseline, Racing, BaseEnv
from modules.models import DQN, A2C, PPO, BaseAgent
from typing import Optional, Type

RACE_VERSION = 0.821


def racing_game(args):
    # assert not (not args.agent_active and not args.agent_train), 'Live agent needs to be active'
    assert not (args.track_cache and args.track_save), 'The track is already cached locally'

    print(f'Running on {RACE_VERSION} ...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current available device is {device.type}')

    # Initialize environment
    envs = []
    if not args.agent_active or not args.agent_train:
        processes_count = 1
    else:
        processes_count = args.processes_count
        assert (processes_count > 0)

    if args.env_name == 'Baseline':
        assert args.agent_active, f'No human interaction on this environment is allowed - {args.env_name}'

        for t in range(processes_count):
            envs.append(Baseline(device, args.frame_size, args.frame_diff))
    elif args.env_name == 'Racing':
        for t in range(processes_count):
            envs.append(
                Racing(device, args.frame_size, args.agent_active, args.track_random_reset,
                       args.track_random_reset_every, args.frame_diff, args.frame_buffer, args.track_cache,
                       args.track_cache_name, args.track_save))

    assert len(envs) > 0, 'Env model is invalid'

    if args.agent_active:
        # Agent's action space
        action_space: int = envs[0].ENV_ACTION_SPACE

        # Initialize the agent
        agent_model: Optional[BaseAgent] = None
        if args.agent_name == 'DQN' or args.agent_name == 'DDQN':
            # DDQN mode
            double = args.agent_name == 'DDQN'

            agent_model = DQN(device, args.frame_shape, action_space, args.agent_cache, args.agent_cache_name,
                              double)
        elif args.agent_name == 'A2C' or args.agent_name == 'A3C':
            # A3C mode
            asynchronous = args.agent_name == 'A3C'

            agent_model = A2C(device, args.frame_shape, action_space, args.agent_cache, args.agent_cache_name,
                              args.model_recurrent, asynchronous, processes_count)
        elif args.agent_name == 'PPO':
            agent_model = PPO(device, args.frame_shape, action_space, args.agent_cache, args.agent_cache_name,
                              args.model_recurrent, processes_count)

        assert agent_model is not None, 'Agent model is invalid'

        if args.agent_train:
            # Model training
            agent_model.model_train(envs, args.episode_count)
        else:
            agent_model.model_valid(envs[0], args.episode_count)
    else:
        envs[0].run()

