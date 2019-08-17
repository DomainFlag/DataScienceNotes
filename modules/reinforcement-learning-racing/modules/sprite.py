import pygame
import numpy as np

TRANSPARENT = (0, 0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class Sprite:

    MAX_VELOCITY: float = 3.45
    MIN_VELOCITY: float = 1.92

    ACTION_SPACE_COUNT = 4

    acceleration: float = 0.025
    steering: float = 0.02
    attenuation: float = 0.25

    def __init__(self, position, rotation, index = 0):
        self.position = position
        self.rotation = rotation
        self.velocity = 0.
        self.index = index

    def initialize(self):
        # Car texture
        self.car_tex = pygame.image.load("./assets/car.png")
        self.car_tex = pygame.transform.scale(self.car_tex, (25, 25))
        self.car_tex.set_colorkey(TRANSPARENT)

        self.car_size_offset = np.array(self.car_tex.get_rect().size) / 2

    def movement(self, acceleration):
        self.velocity += acceleration

        if self.velocity > Sprite.MAX_VELOCITY:
            self.velocity = Sprite.MAX_VELOCITY
        elif self.velocity < -Sprite.MIN_VELOCITY:
            self.velocity = -Sprite.MIN_VELOCITY

    def act(self, scaling):
        direction = np.array([np.cos(self.rotation), -np.sin(self.rotation)])

        self.position += direction * self.velocity * scaling

    def render(self, screen, track, position_offset, hint = True):
        surf = pygame.transform.rotate(self.car_tex, (np.pi / 2.0) / np.pi * 180 + self.rotation / np.pi * 180)
        screen.blit(surf, self.position + position_offset - self.car_size_offset)

        self.index = track.get_sprite_index(self.position, self.index)
        pts, width = track.get_sprite_boundaries(self.position, self.index)
        if hint:
            pygame.draw.circle(screen, GREEN, (pts[0] + track.track_offset).astype(int), 1)
            pygame.draw.circle(screen, GREEN, (pts[1] + track.track_offset).astype(int), 1)

            pygame.draw.circle(screen, RED, (track.get_track_position(self.index) + track.track_offset).astype(int), 2)
