import torch
import numpy as np

from collections import deque


class BaseEnv:

    ENV_ACTION_SPACE: int = 0

    done: bool = False
    exit: bool = False

    states: deque = None
    prev_frame = None
    prev_state, curr_state = None, None

    frame_diff: bool
    frame_pack: bool

    frame_size = None
    frame_shape = None

    def __init__(self, device, frame_shape, frame_diff = False, frame_pack = False):

        self.device = device
        self.frame_shape = np.array(frame_shape)
        if self.frame_shape[0] % 4 == 0:
            self.frame_shape[0] /= 4

        self.frame_size = self.frame_shape[-2:]
        self.frame_diff, self.frame_pack = frame_diff, frame_pack
        self.states = deque([], 4 if frame_pack else 2)

        self.states_fill()

    def states_fill(self):
        for i in range(self.states.maxlen):
            self.states.append(torch.zeros(tuple(self.frame_shape)).to(self.device))

    def init(self):
        raise NotImplementedError

    def step(self, action = None, sync = False):
        raise NotImplementedError

    def event_handler(self):
        pass

    def run(self):
        self.init()
        while not self.exit:
            self.step(action = None, sync = True)

    def reset(self, episode):
        self.done = False
        self.states.clear()
        self.states_fill()
        if self.prev_frame is not None:
            self.prev_frame = None

    def release(self):
        raise NotImplementedError

    def edit_frame(self, frame):
        self.prev_state = self.curr_state

        if self.frame_diff:
            if self.prev_frame is None:
                self.prev_frame = frame

            # Reveal motion
            state = (frame - self.prev_frame).to(self.device)

            self.prev_frame = frame
        else:
            state = frame.to(self.device)

        self.states.append(state)
        if self.frame_pack:
            # Return pack of 4 latest states
            self.curr_state = torch.cat(list(self.states), dim = 0)
        else:
            # Return latest state
            self.curr_state = self.states[-1]

        return self.curr_state
