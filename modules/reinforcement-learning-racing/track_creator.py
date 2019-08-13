import pygame
import random
import os
import numpy as np
import scipy.interpolate as interpolate

from modules import Sprite

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
TRANSPARENT = (0, 0, 0, 0)

ROAD_TRACK = (50, 50, 50)
ROAD_HINT = (143, 152, 86)

FPS_CAP = 60.0
WIDTH = 75

TITLE = "RL racer"
SIZE = [700, 700]


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


def shape_coordinates(pivots, scaling = WIDTH / 2.0):
    if len(pivots) != 3:
        raise Exception("Requires three pivots")

    ab, bc, ac = calc_distance(*pivots[0:2]), calc_distance(*pivots[1:]), calc_distance(pivots[0], pivots[2])
    width, height = (pivots[2] - pivots[1])

    ac1 = (bc ** 2 + ac ** 2 - ab ** 2) / (2 * ac)
    h = np.sqrt(bc ** 2 - ac1 ** 2)
    origin, alpha = np.arctan2(-height, width), np.arctan2(ac1, h)
    angle1, angle2 = origin - alpha, origin + (np.pi - alpha)

    x0 = np.array([np.cos(angle1), np.sin(angle1)]) * scaling + pivots[1]
    x1 = np.array([np.cos(angle2), np.sin(angle2)]) * scaling + pivots[1]

    return [pivots[1], pivots[1], x0, x1], origin, bc


def interpolated_map_generate(pivots, noise = None, knots = 5, order = False):
    if order:
        indices = np.argsort(pivots.T[0])
        xs, ys = pivots[indices].T
    else:
        xs, ys = pivots.T

    xs, ys = np.r_[xs, xs[0]], np.r_[ys, ys[0]]
    weights = noise if noise is not None else None

    tck, u = interpolate.splprep([xs, ys], w = weights, s = 512.0, t = knots, per = True)

    # Evaluate the spline fits for 15000 evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, 15000), tck)

    return xi, yi


def track_line_render(step, offset):
    def renderer(screen, track_offset, xi, yi):
        for index, (x1, y1) in enumerate(zip(xi, yi)):
            ind = (index + 1) % len(xi)

            pos1 = np.array([x1, y1]) + track_offset
            pos2 = np.array([xi[ind], yi[ind]]) + track_offset

            pygame.draw.line(screen, ROAD_TRACK, pos1.astype(int), pos2.astype(int), WIDTH)

            if index % step > offset:
                pygame.draw.line(screen, ROAD_HINT, pos1.astype(int), pos2.astype(int), 3)

    return renderer


def track_texture_render(texture):
    def renderer(screen, track_offset, xi, yi):
        data = np.array([xi, yi]).T
        for index, pivot in enumerate(data):
            adjacent_pivots, _ = pivot_adjacent(data.tolist(), index)
            shape, rotation, distance = shape_coordinates(np.array(adjacent_pivots))

            surf = texture.subsurface((0, 0, 35, 2.0))
            surf = pygame.transform.rotate(surf, rotation * 180 / np.pi)
            screen.blit(surf, pivot + track_offset, (0, 0, 49.5, 49.5))

    return renderer


def create_text_renderer(screen):
    # Load font
    font = pygame.font.SysFont('Comic Sans MS', 24)

    def text_render(text, position):
        surface = font.render(text, False, (0, 0, 0))

        # Render to current surface
        screen.blit(surface, position)

    return text_render


def render_track(pivots, points, xi, yi, track_offset, text_renderer, track_renderer, hints = False):
    # Track surface
    track = pygame.Surface(SIZE)

    # Render hints for track construction
    if hints:
        for point in points:
            pygame.draw.circle(track, BLACK, point + track_offset, 5)

        for index, pivot in enumerate(pivots):
            text_renderer(str(index), pivot + track_offset)

            pygame.draw.circle(track, RED, pivot + track_offset, 5)

    # Render track
    track_renderer(track, track_offset, xi, yi)

    # Optimize drawing
    track.convert()

    return track


def metadata_map_generate(points, index = 0, offset = 100):
    if points is None or len(points) == 0:
        raise Exception("Illegal parameters")

    if index < 0 or index >= len(points):
        raise Exception("Illegal index position")

    curr_pos, next_pos = (points[index], points[index + offset])
    next_pos = np.array(next_pos) - np.array(curr_pos)
    next_pos[-1] *= -1

    angle = np.arctan2(*next_pos[::-1])
    if angle < 0:
        angle = 2 * np.pi - np.abs(angle)

    return curr_pos, angle


def update_index(sprite, points, index):
    while True:
        pivots, indices = pivot_adjacent(points, index)
        distances = [ calc_distance(sprite.position, pivot) for pivot in pivots ]
        max_index = np.argmin(distances)

        if max_index == 0:
            index -= 1
            if index < 0:
                index = len(points) - 1
        elif max_index == 2:
            index += 1
            if index >= len(points):
                index = 0
        else:
            break

    return indices[max_index]


def update_sprite_width(sprite, points, index):
    pivots, _ = pivot_adjacent(points, index)
    shape, rotation, distance = shape_coordinates(pivots)

    dst1 = calc_distance(sprite.position, shape[2])
    dst2 = calc_distance(sprite.position, shape[3])

    return shape[2:], np.minimum(dst1, dst2)


def track_creator(hint = True):
    # Set full screen centered
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    pygame.display.set_caption(TITLE)

    # Set the height and width of the screen
    screen = pygame.display.set_mode(SIZE)

    # Loop until the user clicks the close button.
    done = False

    # Create a text renderer helper function
    text_renderer = create_text_renderer(screen)

    # Generate pivots
    track_offset = (np.mean(SIZE) * 0.1).astype(int)
    track_size = np.array(SIZE) - track_offset * 2.0

    points, pivots = pivot_map_generate(32, *track_size)
    noise = np.random.random(len(pivots) + 1) * 50.03

    # Create the track and renderer, render to static surface
    xi, yi = interpolated_map_generate(np.array(pivots), noise = noise)
    track_data = np.array([xi, yi]).T
    track_renderer = track_line_render(156, 48)
    track = render_track(pivots, points, xi, yi, track_offset, text_renderer, track_renderer)

    # Road texture
    road_tex = pygame.image.load("./assets/tunnel_road.jpg")
    road_tex = pygame.transform.scale(road_tex, (35, 35))
    road_tex.set_colorkey(TRANSPARENT)

    # Car texture
    car_tex = pygame.image.load("./assets/car.png")
    car_tex = pygame.transform.scale(car_tex, (25, 25))
    car_tex.set_colorkey(TRANSPARENT)

    car_size_offset = np.array(car_tex.get_rect().size) / 2

    # Generate track meta-data
    sprite_index = 0
    start_pos, start_rot = metadata_map_generate(track_data, index = sprite_index)

    # Create a sprite given meta-data
    sprite = Sprite(np.array(start_pos), start_rot)

    # Set up timer for smooth rendering
    clock = pygame.time.Clock()

    while not done:

        # Continuous key press
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            sprite.movement(Sprite.acceleration)
        elif keys[pygame.K_DOWN]:
            sprite.movement(-Sprite.acceleration)

        if keys[pygame.K_LEFT]:
            sprite.rotation += sprite.steering
        elif keys[pygame.K_RIGHT]:
            sprite.rotation -= sprite.steering

        # User did something
        for event in pygame.event.get():
            # Close button is clicked
            if event.type == pygame.QUIT:
                done = True

            # Escape key is pressed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

        # Clear the screen and set the screen background
        screen.fill(WHITE)

        # Render track
        screen.blit(track, (0, 0))

        # Render objects
        surf = pygame.transform.rotate(car_tex, (np.pi / 2.0) / np.pi * 180 + sprite.rotation / np.pi * 180)
        screen.blit(surf, sprite.position + track_offset - car_size_offset)

        # Activate any activable units
        sprite.act()
        sprite_index = update_index(sprite, track_data, sprite_index)
        pts, width = update_sprite_width(sprite, track_data, sprite_index)
        if hint:
            pygame.draw.circle(screen, GREEN, (pts[0] + track_offset).astype(int), 1)
            pygame.draw.circle(screen, GREEN, (pts[1] + track_offset).astype(int), 1)

            pygame.draw.circle(screen, RED, (track_data[sprite_index] + track_offset).astype(int), 2)

        # Update the screen
        pygame.display.flip()

        # Handle constant FPS cap
        pygame.display.set_caption("{0}: {1:.2f}".format(TITLE, clock.get_fps()))

        clock.tick(FPS_CAP)

    # Be IDLE friendly
    pygame.quit()
