import pygame
import random
import numpy as np
import scipy.interpolate as interpolate

from math import cos, sin
from pygame import gfxdraw
from modules import Sprite

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
TRANSPARENT = (0, 0, 0, 0)

ROAD_TRACK = (50, 50, 50)
ROAD_HINT = (143, 152, 86)


class Track:

    WIDTH_MAX: int = 60

    sprite: Sprite = None
    pts: list
    lap: int = 0
    width: float = 0.
    index: int = 0

    static_params: dict = {
        "width_max": WIDTH_MAX
    }

    def initialize(self, size, text_renderer):
        # Generate pivots
        self.track_offset = (np.mean(size) * 0.1).astype(int)
        track_size = np.array(size) - self.track_offset * 2.0

        points, pivots = pivot_map_generate(32, *track_size)
        noise = np.random.random(len(pivots) + 1) * 50.03

        # Create the track and renderer, render to static surface
        xi, yi = interpolated_map_generate(np.array(pivots), noise = noise)
        self.track_data = np.array([xi, yi]).T
        self.track = render_track(size, pivots, points, xi, yi, self.track_offset, text_renderer, track_line_render(320, 175))

        # Create the sprite
        start_pos, start_rot = self.get_metadata(index=0)

        self.sprite = Sprite(np.array(start_pos), start_rot, self.track_offset)
        self.sprite.initialize()

        # Road texture
        road_tex = pygame.image.load("./assets/road.jpg")
        road_tex = pygame.transform.scale(road_tex, (35, 35))
        road_tex.set_colorkey(TRANSPARENT)

    def get_sprite_index(self, position, index):
        while True:
            pivots, indices = pivot_adjacent(self.track_data, index)
            distances = [ calc_distance(position, pivot) for pivot in pivots ]
            max_index = np.argmin(distances)

            if max_index == 0:
                index -= 1
                if index < 0:
                    index = len(self.track_data) - 1
            elif max_index == 2:
                index += 1
                if index >= len(self.track_data):
                    index = 0
            else:
                break

        return indices[max_index]

    def get_sprite_boundaries(self, position, index):
        pivots, _ = pivot_adjacent(self.track_data, index)
        shape, rotation, distance = shape_coordinates(pivots)

        dsts = [ calc_distance(position, pt) for pt in shape ]

        return shape, np.min(dsts)

    def get_track_position(self, index):
        return self.track_data[index].copy()

    def get_metadata(self, index = 0, offset = 100):
        if self.track_data is None or len(self.track_data) == 0:
            raise Exception("Illegal parameters")

        if index < 0 or index >= len(self.track_data):
            raise Exception("Illegal index position")

        curr_pos, next_pos = (self.track_data[index].copy(), self.track_data[index + offset].copy())
        next_pos = np.array(next_pos) - np.array(curr_pos)
        next_pos[-1] *= -1

        angle = np.arctan2(*next_pos[::-1])
        if angle < 0:
            angle = 2 * np.pi - np.abs(angle)

        return curr_pos, angle

    def reset_track(self):
        # Reset the sprite
        self.sprite.reset()

        start_pos, start_rot = self.get_metadata(index = 0)
        self.sprite.position, self.sprite.rotation = start_pos, start_rot
        self.index, self.lap = 0, 0

    def act(self, scaling):
        self.sprite.act(scaling)

        curr_index = self.get_sprite_index(self.sprite.position, self.index)
        if curr_index < self.index:
            self.lap += 1

        self.index = curr_index
        self.pts, self.width = self.get_sprite_boundaries(self.sprite.position, self.index)

    def render(self, screen, hint = True):
        # Render track
        screen.blit(self.track, (0, 0))

        if hint:
            for pt in self.pts:
                pygame.draw.circle(screen, GREEN, (pt + self.track_offset).astype(int), 1)

            pygame.draw.circle(screen, RED, (self.get_track_position(self.index) + self.track_offset).astype(int), 2)

        # Render the sprites
        self.sprite.render(screen)

    def get_params(self):
        params = Track.static_params.copy()
        params.update({
            "width": self.width,
            "index": self.index,
            "alive": True
        })

        params.update(self.sprite.get_params())

        return params


def get_angle(p0, p1, p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))

    return np.degrees(angle)


def point_generate(history, width, height):
    while True:
        x = random.randint(0, width)

        if x not in history:
            history.add(x)
            break

    y = random.randint(0, height)

    return x, y


def calc_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def calc_point_distance(center = None):
    return lambda point: calc_distance(point, center)


def point_angle(point, width, height):
    pivot_transl = np.array(point) - np.array([width, height]) / 2.0
    pivot_normal = pivot_transl / np.sqrt(np.dot(pivot_transl, pivot_transl))

    angle = np.arctan2(pivot_normal[1], pivot_normal[0])
    if angle < 0:
        angle = 2 * np.pi - np.abs(angle)

    return angle


def pivot_map_generate(size, width, height, threshold_dist = 0.75, threshold_angle = 0.3):
    history = set()
    points = [ point_generate(history, width, height) for _ in range(size) ]

    angles = {}

    pivot_angle = None
    pivot_distance = 0
    for point in points:
        angle = point_angle(point, width, height)
        angles[angle] = point

        point_distance = np.linalg.norm(point)
        if point_distance > pivot_distance:
            pivot_angle, pivot_distance = angle, point_distance

    angles_sorted = sorted(angles)

    # Extract the pivot; starting point
    pivot_index = angles_sorted.index(pivot_angle)

    # Distance calculator from the origin
    calc_point_dist = calc_point_distance(np.array([width, height]) / 2.0)

    pivots = [ angles[angles_sorted[pivot_index]] ]
    for index in range(pivot_index + 1, len(angles_sorted) + pivot_index):
        angle = angles_sorted[index % len(angles_sorted)]

        # Compute the proportionality between pivot and possible adjacent pivot candidate
        prop = np.abs(calc_point_dist(pivots[-1]) / calc_point_dist(angles[angle]) - 1.0)

        if np.abs(pivot_angle - angle) > threshold_angle and prop < threshold_dist:
            pivots.append(angles[angle])

            pivot_angle = angle

    return points, pivots


def pivot_adjacent(pivots, index):
    if index == 0:
        return np.concatenate(([pivots[-1]], pivots[index:index + 2])), [len(pivots) - 1, index, index + 1]
    elif index == len(pivots) - 1:
        return np.concatenate((pivots[index - 1:index + 1], [pivots[0]])), [index - 1, index, 0]
    else:
        return np.array(pivots[index - 1:index + 2]), list(range(index - 1, index + 2))


def shape_coordinates(pivots, scaling = Track.WIDTH_MAX / 2.0):
    if len(pivots) != 3:
        raise Exception("Requires three pivots")

    ab, bc, ac = calc_distance(*pivots[0:2]), calc_distance(*pivots[1:]), calc_distance(pivots[0], pivots[2])
    width, height = (pivots[2] - pivots[1])

    ac1 = (bc ** 2 + ac ** 2 - ab ** 2) / (2 * ac)
    orientation, alpha = np.arctan2(height, width), np.arcsin(ac1 / bc)
    angle1, angle2 = orientation - alpha, orientation + (np.pi - alpha)

    x0 = np.array([np.cos(angle1), np.sin(angle1)]) * scaling + pivots[1]
    x1 = np.array([np.cos(angle2), np.sin(angle2)]) * scaling + pivots[1]

    return [x0, x1], orientation, bc


def interpolated_map_generate(pivots, noise = None, knots = 3, order = False, precision = 20000):
    if order:
        indices = np.argsort(pivots.T[0])
        xs, ys = pivots[indices].T
    else:
        xs, ys = pivots.T

    xs, ys = np.r_[xs, xs[0]], np.r_[ys, ys[0]]
    weights = noise if noise is not None else None

    tck, u = interpolate.splprep([xs, ys], w = weights, s = 0, t = knots, per = True)

    # Evaluate the spline fits for precision* evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, precision), tck)

    return xi, yi


def draw_aaline(surface, line, color, width = 1.0, length = None):
    center = np.mean(line, axis = 0)

    if length is None:
        length = calc_distance(line[0], line[1])

    angle = np.arctan2(line[0][1] - line[1][1], line[0][0] - line[1][0])
    direction = np.array([np.cos(angle), np.sin(angle)])

    top_left = (+(length / 2.) * direction[0] - (width / 2.) * direction[1],
        +(width / 2.) * direction[0] + (length / 2.) * direction[1])
    top_right = (-(length / 2.) * direction[0] - (width / 2.) * direction[1],
        +(width / 2.) * direction[0] - (length / 2.) * direction[1])
    bottom_left = (+(length / 2.) * direction[0] + (width / 2.) * direction[1],
        -(width / 2.) * direction[0] + (length / 2.) * direction[1])
    bottom_right = (-(length / 2.) * direction[0] + (width / 2.) * direction[1],
        -(width / 2.) * direction[0] - (length / 2.) * direction[1])

    points = np.array([top_left, top_right, bottom_left, bottom_right]) + center

    pygame.gfxdraw.aapolygon(surface, points, color)
    pygame.gfxdraw.filled_polygon(surface, points, color)


def track_line_render(step, offset):
    def renderer(screen, track_offset, xi, yi):
        for index, (x1, y1) in enumerate(zip(xi, yi)):
            ind = (index + 1) % len(xi)

            pos1 = np.array([x1, y1]) + track_offset
            pos2 = np.array([xi[ind], yi[ind]]) + track_offset

            draw_aaline(screen, [pos1, pos2], ROAD_TRACK, width = Track.WIDTH_MAX)

            if index % step > offset:
                draw_aaline(screen, [pos1, pos2], ROAD_HINT, width = 2, length = 4.0)

    return renderer


def track_texture_render(texture):
    # TODO(1) Rework needed
    def renderer(screen, track_offset, xi, yi):
        data = np.array([xi, yi]).T
        for index, pivot in enumerate(data):
            adjacent_pivots, _ = pivot_adjacent(data.tolist(), index)
            shape, rotation, distance = shape_coordinates(np.array(adjacent_pivots))

            surf = texture.subsurface((0, 0, 35, 2.0))
            surf = pygame.transform.rotate(surf, rotation * 180 / np.pi)
            screen.blit(surf, pivot + track_offset, (0, 0, Track.WIDTH_MAX, Track.WIDTH_MAX))

    return renderer


def render_track(size, pivots, points, xi, yi, track_offset, text_renderer, track_renderer, hints = False):
    # Track surface
    track = pygame.Surface(size)

    # Render track
    track_renderer(track, track_offset, xi, yi)

    # Render hints for track construction
    if hints:
        for point in points:
            pygame.draw.circle(track, WHITE, point + track_offset, 5)

        for index, pivot in enumerate(pivots):
            text_renderer(str(index), pivot + track_offset)

            pygame.draw.circle(track, RED, pivot + track_offset, 5)

    # Optimize drawing
    track.convert()

    return track
