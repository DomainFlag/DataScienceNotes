import argparse

from racing_game import racing_game

# Available choices
ENV_OPTIONS = ['Baseline', 'Racing']
AGENT_OPTIONS = ['DQN', 'DDQN', 'A2C', 'A3C', 'PPO']

# Command-line arguments
parser = argparse.ArgumentParser(description = 'Reinforcement Learning & Racing')

parser.add_argument('--env-name', default = ENV_OPTIONS[1], const = ENV_OPTIONS[1], nargs = '?', metavar = 'ENV',
                    help = f"environment to train on (default: {ENV_OPTIONS[1]}) (choices: {', '.join(ENV_OPTIONS)})",
                    choices = ENV_OPTIONS)
parser.add_argument('--agent-name', default = AGENT_OPTIONS[2], const = AGENT_OPTIONS[2], nargs = '?', metavar = 'AGENT',
                    help = f"agent model to be trained (default: {AGENT_OPTIONS[2]}) (choices: {', '.join(AGENT_OPTIONS)})",
                    choices = AGENT_OPTIONS)

parser.add_argument('--agent-active', action = 'store_true', default = False,
                    help = 'agent or human to act on env (default: False)')
parser.add_argument('--agent-train', action = 'store_true', default = False,
                    help = 'agent to act/train on env (default: False)')

parser.add_argument('--episode-count', type = int, default = 150, metavar = 'EC',
                    help = 'episode count the model will be trained on (default: 300)')
parser.add_argument('--processes-count', type = int, default = 4, metavar = 'PC',
                    help = 'processes count the model will be trained in parallel (default: 4)')

parser.add_argument('--agent-cache', action = 'store_true', default = False,
                    help = 'agent model to be loaded (default: False)')
parser.add_argument('--agent-cache-name', type = str, default = 'model.pt', metavar = 'ACL',
                    help = 'agent model location relative to static folder (default: model.pt)')
parser.add_argument('--agent-interactive', action = 'store_true', default = False,
                    help = 'display cached agent meta-data (default: False)')

parser.add_argument('--model-recurrent', action = 'store_true', default = False,
                    help = 'use RNN for agent model (default: False)')

parser.add_argument('--frame-size', nargs = '*', default = [150, 150], metavar = 'FS',
                    help = 'train sample frame size (default: [150, 150])')
parser.add_argument('--frame-diff', action = 'store_true', default = False,
                    help = 'use the difference between each two subsequent frames (default: False)')
parser.add_argument('--frame-pack', action = 'store_true', default = False,
                    help = 'stack 4 latest frames as 4 channels (default: False)')
parser.add_argument('--frame-buffer', action = 'store_true', default = False,
                    help = 'no window will be displayed (default: False)')
parser.add_argument('--frame-grayscale', action = 'store_true', default = False,
                    help = 'use grayscale image samples (default: False)')

parser.add_argument('--track-save', action = 'store_true', default = False,
                    help = 'track to be saved locally (default: False)')
parser.add_argument('--track-cache', action = 'store_true', default = False,
                    help = 'track to be loaded instead of being generated (default: False)')
parser.add_argument('--track-cache-name', type = str, default = 'track_model.npy', metavar = 'TCL',
                    help = 'track data location relative to static folder (default: track_model.npy)')
parser.add_argument('--track-random-reset', action = 'store_true', default = False,
                    help = 'random reset the actor in env (default: False)')
parser.add_argument('--track-random-reset-every', type = int, default = 6, metavar = 'RRE',
                    help = 'random reset the actor every number of times (default: 6)')

if __name__ == '__main__':
    args = parser.parse_args()
    channels = 1 if args.frame_grayscale else 3
    if args.frame_pack:
        channels *= 4

    args.frame_shape = (channels, *args.frame_size)

    racing_game(args)
