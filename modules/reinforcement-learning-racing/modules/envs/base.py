class BaseEnv:

    ENV_ACTION_SPACE: int = 0

    done: bool = False
    exit: bool = False

    prev_state, curr_state = None, None
    prev_frame = None

    frame_diff: bool

    def __init__(self, device, frame_diff = True):

        self.device = device
        self.frame_diff = frame_diff

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
        if self.prev_frame is not None:
            self.prev_frame = None

    def release(self):
        raise NotImplementedError

    def edit_frame(self, frame):
        if self.frame_diff:
            if self.prev_frame is None:
                self.prev_frame = frame

            state = (frame - self.prev_frame).to(self.device)

            self.prev_frame = frame
        else:
            state = frame.to(self.device)

        self.prev_state = self.curr_state
        self.curr_state = state

        return state
