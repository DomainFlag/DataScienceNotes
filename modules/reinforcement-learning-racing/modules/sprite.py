import pygame
import numpy as np

TRANSPARENT = (0, 0, 0, 0)
RED = (255, 0, 0)


class Sprite:

    SPRITE_SIZE: np.ndarray = np.array([25, 25])

    MAX_VELOCITY: float = 2.15
    MIN_VELOCITY: float = 1.4

    ACTION_SPACE_COUNT = 2
    MOTION_SPACE_COUNT = 2
    STEERING_SPACE_COUNT = 2

    acceleration: float = 0.05
    steering: float = 0.1
    attenuation: float = 0.25

    static_params: dict = {
        "acc_max": MAX_VELOCITY,
        "acc_min": MIN_VELOCITY,
        "acc_amount": acceleration,
        "steering_amount": steering
    }

    def __init__(self, position, rotation, offset):
        self.position = position
        self.rotation = rotation
        self.offset = offset
        self.velocity = Sprite.MIN_VELOCITY

    def initialize(self):
        # Car texture
        self.car_tex = pygame.image.load("./assets/car.png")
        self.car_tex = pygame.transform.scale(self.car_tex, Sprite.SPRITE_SIZE)
        self.car_tex.set_colorkey(TRANSPARENT)
        self.car_size = np.array(self.car_tex.get_rect().size)
        self.car_size_offset = self.car_size / 2

    def steer(self, steering):
        self.rotation += steering

        if self.rotation >= 2 * np.pi:
            self.rotation -= 2 * np.pi
        elif self.rotation < 0:
            self.rotation = 2 * np.pi + self.rotation

    def movement(self, acceleration):
        self.velocity += acceleration

        if self.velocity > Sprite.MAX_VELOCITY:
            self.velocity = Sprite.MAX_VELOCITY
        elif self.velocity < Sprite.MIN_VELOCITY:
            self.velocity = Sprite.MIN_VELOCITY

    def act(self, scaling = 1.0):
        direction = np.array([np.cos(self.rotation), -np.sin(self.rotation)])

        self.position += direction * self.velocity * scaling

    def act_action(self, action):
        """ action-4 do nothing """
        if action is None:
            return None

        motion, steering = 0., 0.
        if action == 0:
            motion += Sprite.acceleration

        if action == 1:
            motion -= Sprite.acceleration

        if action == 2:
            steering += Sprite.steering

        if action == 3:
            steering -= Sprite.steering

        self.movement(motion)
        self.steer(steering)

    def reset(self):
        self.velocity = Sprite.MIN_VELOCITY
        self.rotation = 0.

    def get_position(self):
        return self.position + self.offset

    def get_sprite_rotated(self, img, angle, offset):
        """ Rotate the image while keeping its center. """

        rot_image = pygame.transform.rotate(img, angle)
        rot_rect = rot_image.get_rect(center = img.get_rect().center)

        return rot_image, rot_rect.move(*offset)

    def render(self, screen):
        # Get params
        offset = self.position + self.offset - self.car_size_offset
        rotation = 90 + self.rotation / np.pi * 180

        # Rotate an image from its center
        surf, rect = self.get_sprite_rotated(self.car_tex, rotation, offset)

        screen.blit(surf, rect)

    def get_params(self):
        params = Sprite.static_params.copy()
        params.update({
            "pos": self.get_position(),
            "acc": self.velocity,
            "rot": self.rotation
        })

        return params
