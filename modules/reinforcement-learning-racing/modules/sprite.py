import numpy as np


class Sprite:

    MAX_VELOCITY: float = 3.45

    acceleration: float = 0.025
    steering: float = 0.02
    attenuation: float = 0.25

    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation
        self.velocity = 0.

    def movement(self, acceleration):
        self.velocity += acceleration

        if self.velocity > Sprite.MAX_VELOCITY:
            self.velocity = Sprite.MAX_VELOCITY
        elif self.velocity < 0:
            self.velocity = 0

    def act(self):
        direction = np.array([np.cos(self.rotation), -np.sin(self.rotation)])

        self.position += direction * self.velocity
