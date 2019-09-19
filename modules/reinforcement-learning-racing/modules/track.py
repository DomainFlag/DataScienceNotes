import pygame
import random
import numpy as np
import scipy.interpolate as interpolate

from pygame import gfxdraw
from modules import Sprite

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
TRANSPARENT = (0, 0, 0, 0)

ROAD_TRACK = (50, 50, 50)
ROAD_HINT = (143, 152, 86)


class Track:

    TRACK_PRECISION: int = 5000
    TRACK_OFFSET: int = TRACK_PRECISION // 10
    WIDTH_MAX: int = 60

    points: list
    pivots: list

    sprite: Sprite = None
    pts: list = None
    lap: int = 0
    width: float = 0.
    start_index: int = -1
    index: int = 0
    progress: int = 0
    progress_max: int = 0

    static_params: dict = {
        "width_max": WIDTH_MAX,
        "width_half": WIDTH_MAX // 2,
        "index_max": TRACK_PRECISION
    }

    def __init__(self):
        # Road texture
        road_tex = pygame.image.load("./assets/road.jpg")
        road_tex = pygame.transform.scale(road_tex, (35, 35))
        road_tex.set_colorkey(TRANSPARENT)

    def initialize_track(self, size, text_renderer = None, track_save = False, track_cache = False, filename = None):
        self.size, self.track_offset = size, (np.mean(size) * 0.1).astype(int)
        track_size = np.array(size) - self.track_offset * 2.0

        if track_cache:
            self.load_track_from_dict(filename)
        else:
            # Generate pivots
            self.points, self.pivots = pivot_map_generate(32, *track_size)
            self.noise_seed = np.random.randint(np.iinfo(np.int32).max)

            np.random.seed(seed = self.noise_seed)
            self.noise_data = np.random.random(len(self.pivots) + 1) * 50.03

            # Create the track and renderer, render to static surface
            xi, yi = interpolated_map_generate(np.array(self.pivots), noise = self.noise_data)
            self.track_data = np.array([xi, yi]).T

        self.track_surface = render_track(size, self.pivots, self.points, self.track_data, self.track_offset,
                                          text_renderer, track_line_render(320, 175))

        if track_save:
            # Save track data for inference and model validation
            self.save_track_to_dict(filename = "track_model.npy" if filename is None else filename)

    def initialize_sprite(self, index = 0):
        self.start_index = index

        # Create the sprite
        start_pos, start_rot = self.get_metadata(index = index)

        self.sprite = Sprite(np.array(start_pos), start_rot, self.track_offset)
        self.sprite.initialize()

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

        return shape, np.min(dsts), rotation

    def get_track_position(self, index):
        return self.track_data[index].copy() + self.track_offset

    def get_metadata(self, index = 0, offset = 25):
        if self.track_data is None or len(self.track_data) == 0:
            raise Exception("Illegal parameters")

        if index < 0 or index >= len(self.track_data):
            raise Exception("Illegal index position")

        curr_pos, next_pos = (self.track_data[index].copy(), self.track_data[(index + offset) % len(self.track_data)].copy())
        next_pos = np.array(next_pos) - np.array(curr_pos)
        next_pos[-1] *= -1

        angle = np.arctan2(*next_pos[::-1])
        if angle < 0:
            angle = 2 * np.pi - np.abs(angle)

        return curr_pos, angle

    def reset_track(self, random_reset = False):
        # Reset the sprite
        self.sprite.reset()

        # Random reset position
        self.start_index = 0 if not random_reset else np.random.randint(len(self.track_data))

        start_pos, start_rot = self.get_metadata(index = self.start_index)
        self.sprite.position, self.sprite.rotation = start_pos, start_rot
        self.index, self.lap = self.start_index, 0
        self.progress, self.progress_max = 0, 0

    def is_to_left(self):
        """ Helper function for retrieving whenever the sprite is to the left of track from track perspective """
        pos_origin, pos_track = self.sprite.get_position(), self.get_track_position(self.index)
        origin_angle, magnitude = point_angle(pos_origin, pos_track, invert = True)
        if origin_angle is None:
            return None

        offset_angle = self.get_metadata(self.index, offset = 5)[-1]
        origin_angle -= offset_angle

        pos_translated = np.array([np.cos(origin_angle), np.sin(origin_angle)]) * magnitude + pos_track
        return pos_translated[1] > pos_track[1]

    def is_alive(self, state = None, centered = False, precision = True):
        if precision and state is not None:
            size, rot = self.sprite.SPRITE_SIZE, self.sprite.rotation
            if not centered:
                position = self.sprite.get_position()
            else:
                position = np.array(state.size) / 2

            for x in [0, size[0]]:
                for y in range(0, size[1]):
                    alive = self.check_alive(state, position, x, y, size, rot) and \
                            self.check_alive(state, position, y, x, size, rot)

                    if not alive:
                        return False

            return True
        else:
            if self.pts is None:
                self.pts, self.width, self.rot = self.get_sprite_boundaries(self.sprite.position, self.index)

            pt1, pt2 = (self.pts[0], self.pts[1]) if self.pts[0][0] < self.pts[1][0] else (self.pts[1], self.pts[0])
            norm_pt = pt2 - pt1
            norm_pt = norm_pt / np.linalg.norm(norm_pt)

            angle = np.arctan2(norm_pt[1], norm_pt[0])

            norm_target_pt = self.sprite.position - pt1
            scale = np.linalg.norm(norm_target_pt)
            norm_target_pt /= scale

            angle_target = np.arctan2(norm_target_pt[1], norm_target_pt[0]) - angle
            transl_target_pt = np.array([np.cos(angle_target), np.sin(angle_target)]) * scale + pt1

            x0, x1, x2 = pt1[0], transl_target_pt[0], pt1[0] + calc_distance(self.pts[0], self.pts[1])

            return np.isnan([x0, x1, x2]).any() or (x0 <= x1 <= x2)

    def act(self, scaling):
        self.sprite.act(scaling)

        index = self.get_sprite_index(self.sprite.position, self.index)

        self.progress += Track.get_index_offset(self.index, index)
        self.index = index
        self.pts, self.width, self.rot = self.get_sprite_boundaries(self.sprite.position, self.index)

        if self.progress >= Track.TRACK_PRECISION:
            self.progress %= Track.TRACK_PRECISION
            self.lap += 1
        elif self.progress < 0:
            self.progress = Track.TRACK_PRECISION + self.progress
            self.lap -= 1

        self.progress_max = max(self.lap * Track.TRACK_PRECISION + self.progress, self.progress_max)

    def render(self, screen, direction_offset = 200, hint_direction = True, hint_boundary = True):
        # Render track
        screen.blit(self.track_surface, (0, 0))

        # Render the sprites
        self.sprite.render(screen)

        if hint_direction:
            position_hint = self.track_data[(self.index + direction_offset) % len(self.track_data)] + self.track_offset

            pygame.draw.circle(screen, YELLOW, position_hint.astype(int), 5)

        if hint_boundary:
            for pt in self.pts:
                pygame.draw.circle(screen, GREEN, (pt + self.track_offset).astype(int), 1)

            pygame.draw.circle(screen, RED, self.get_track_position(self.index).astype(int), 2)

    def get_params(self, state = None, centered = False):
        params = Track.static_params.copy()
        params.update({
            "width": self.width,
            "index": self.index,
            "index_pos": self.get_track_position(self.index),
            "is_to_left": self.is_to_left(),
            "start_index": self.start_index,
            "lap": self.lap,
            "progress": (self.progress, self.progress / self.TRACK_PRECISION * 100),
            "progress_total": self.lap * Track.TRACK_PRECISION + self.progress,
            "progress_max": self.progress_max,
            "alive": self.is_alive(state, centered),
            "angle": self.get_metadata(self.index, offset = 5)[-1]
        })

        params.update(self.sprite.get_params())

        return params

    def load_track_from_dict(self, filename):
        """ Load track data from dict """

        # load track data
        track_dict = np.load("./static/" + filename, allow_pickle = True).item()

        assert(isinstance(track_dict, dict))

        self.points, self.pivots, self.size, self.track_data = track_dict["points"], track_dict["pivots"], \
            track_dict["size"], track_dict["data"]

    def save_track_to_dict(self, filename):
        """ Save track data internally """

        # save track data
        track_dict = {
            'size': self.size,
            'points': self.points,
            'pivots': self.pivots,
            'seed': self.noise_seed,
            'data': self.track_data
        }

        np.save("./static/" + filename, track_dict)

        print("Track saved successfully")

    @staticmethod
    def get_index_offset(index1, index2):
        if abs(index1 - index2) > Track.TRACK_OFFSET:
            # The sprite is located between the start/end extremity
            index_lower, index_upper = (index2, index1) if (index1 > index2) else (index1, index2)

            offset = index_lower - (index_upper - Track.TRACK_PRECISION)

            return offset if index2 < Track.TRACK_OFFSET else -offset
        else:
            return index2 - index1

    @staticmethod
    def check_alive(state, origin, x, y, size, rot):
        coord = np.array([x, y]) - size / 2
        coord_rot = np.arctan2(coord[1], coord[0]) - rot
        coord = np.array([np.cos(coord_rot), np.sin(coord_rot)]) * np.sqrt(np.dot(coord, coord))
        coord += origin + size / 2

        if state.getpixel((int(coord[0]), int(coord[1]))) == BLACK:
            return False

        return True


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


def point_angle(point, origin, invert = False):
    if (point == origin).all():
        return None, None

    pivot_transl = point - origin
    if invert:
        pivot_transl[-1] *= -1

    pivot_distance = np.sqrt(np.dot(pivot_transl, pivot_transl))
    pivot_normal = pivot_transl / pivot_distance

    angle = np.arctan2(pivot_normal[1], pivot_normal[0])
    if angle < 0:
        angle = 2 * np.pi - np.abs(angle)

    return angle, pivot_distance


def pivot_map_generate(size, width, height, threshold_dist = 0.75, threshold_angle = 0.3):
    history = set()
    points = [ point_generate(history, width, height) for _ in range(size) ]

    angles = {}

    pivot_angle = None
    pivot_distance = 0
    for point in points:
        angle, _ = point_angle(np.array(point), np.array([width, height]) / 2.0, invert = True)
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


def pivot_adjacent(pivots, index, boundary = True):
    indices = [ x % len(pivots) for x in range(index, index + 3) ]
    points = [ pivots[x % len(pivots)] for x in indices if not boundary or not x == 0 ]

    if boundary and 0 in indices:
        points.append(pivots[(indices[-1] + 1) % len(pivots)])

    return np.array(points), indices


def shape_coordinates(points, scaling = Track.WIDTH_MAX / 2.0):
    if len(points) != 3:
        raise Exception("Requires three points")

    ab, bc, ac = calc_distance(*points[0:2]), calc_distance(*points[1:]), calc_distance(points[0], points[2])
    width, height = (points[2] - points[1])

    ac1 = (bc ** 2 + ac ** 2 - ab ** 2) / (2 * ac)
    orientation, alpha = np.arctan2(height, width), np.arcsin(ac1 / bc)
    angle1, angle2 = orientation - alpha, orientation + (np.pi - alpha)

    x0 = np.array([np.cos(angle1), np.sin(angle1)]) * scaling + points[1]
    x1 = np.array([np.cos(angle2), np.sin(angle2)]) * scaling + points[1]

    return [x0, x1], orientation, bc


def interpolated_map_generate(pivots, noise = None, knots = 3, order = False, precision = Track.TRACK_PRECISION):
    if order:
        indices = np.argsort(pivots.T[0])
        xs, ys = pivots[indices].T
    else:
        xs, ys = pivots.T

    xs, ys = np.r_[xs, xs[0]], np.r_[ys, ys[0]]
    weights = noise if noise is not None else None

    # Create the B-spline representation that fits most the data points
    tck, u = interpolate.splprep([xs, ys], w = weights, s = 0, t = knots, per = True)

    # Evaluate the spline fits for precision* evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, precision), tck)

    return xi, yi


def draw_line_aliased(surface, line, color, width = 1.0, length = None):
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
    def renderer(screen, track_offset, track_data):
        pos2 = None
        for index, pos1 in enumerate(track_data):
            pos1 = pos1.copy() + track_offset

            if pos2 is not None:
                draw_line_aliased(screen, [pos1, pos2], ROAD_TRACK, width = Track.WIDTH_MAX)

                if index % step > offset:
                    draw_line_aliased(screen, [pos1, pos2], ROAD_HINT, width = 2, length = 4.0)

            pos2 = pos1

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


def render_track(size, pivots, points, track_data, track_offset, text_renderer, track_renderer, hints = False):
    # Track surface
    track = pygame.Surface(size)

    # Render track
    track_renderer(track, track_offset, track_data)

    # Render hints for track construction
    if hints:
        for point in points:
            pygame.draw.circle(track, WHITE, point + track_offset, 5)

        if text_renderer is not None:
            for index, pivot in enumerate(pivots):
                text_renderer(str(index), pivot + track_offset)

                pygame.draw.circle(track, RED, pivot + track_offset, 5)

    return track
