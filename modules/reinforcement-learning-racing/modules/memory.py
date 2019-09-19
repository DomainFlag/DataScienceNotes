import random

from collections import namedtuple
from typing import List, Optional

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    MEMORY_CAPACITY: int = 15000
    REPLAY_START: int = 3250

    capacity: int
    memory: List[Transition]
    position: int
    reward_acc: float = 0.

    def __init__(self, capacity: int = None):
        self.memory = []
        self.position = 0
        self.capacity = capacity if capacity is not None else ReplayMemory.MEMORY_CAPACITY

    def push(self, transition: Transition):
        """ Save transition by possibly overriding pre-existing transitions """
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity
        self.reward_acc += getattr(transition, "reward").item()

    def sample_transition_batch(self, batch_size) -> Optional[Transition]:
        transitions = random.sample(self.memory, batch_size)

        return Transition(*zip(*transitions))

    def reset_history(self):
        self.reward_acc = 0.

    def is_ready(self):
        return self.REPLAY_START <= len(self.memory)

    def __len__(self):
        return len(self.memory)