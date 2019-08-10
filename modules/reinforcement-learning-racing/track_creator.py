import pygame
import random
import os
import numpy as np
import scipy.interpolate as interpolate

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

SCALE = 20.0


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
        return [pivots[-1]] + pivots[index:index + 2]
    elif index == len(pivots) - 1:
        return pivots[index - 1:index + 1] + [pivots[0]]
    else:
        return pivots[index - 1:index + 2]


def shape_coordinates(pivots, scaling = SCALE):
    if len(pivots) != 3:
        raise Exception("Requires three pivots")

    ab, bc, ac = calc_distance(pivots[0], pivots[1]), calc_distance(pivots[1], pivots[2]), calc_distance(pivots[0], pivots[2])
    width, height = (pivots[2] - pivots[1])

    ac1 = (bc ** 2 + ac ** 2 - ab ** 2) / (2 * ac)
    origin, alpha = np.arctan2(height, width), np.arcsin(ac1 / bc)
    angle1, angle2 = origin - alpha, origin + (np.pi - alpha)

    x0 = np.array(np.cos(angle1), np.sin(angle1)) * scaling + pivots[1]
    x1 = np.array(np.cos(angle2), np.sin(angle2)) * scaling + pivots[1]

    return [pivots[1], pivots[1], x0, x1], origin


def interpolated_map_generate(screen, texture, pivots, track_offset, noise = None, knots = 3, line_mode = False):
    indices = np.argsort(pivots.T[0])
    xs, ys = pivots[indices].T

    xs, ys = np.r_[xs, xs[0]], np.r_[ys, ys[0]]
    weights = noise if noise is not None else None

    tck, u = interpolate.splprep([xs, ys], w = weights, s = 1500.0, t = knots, per = True)

    # Evaluate the spline fits for 2000 evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, 2000), tck)

    # Draw line mode
    if line_mode:
        for index, x in enumerate(xi):
            pos = np.array([x, yi[index]]) + track_offset

            pygame.draw.circle(screen, GREEN, pos.astype(int), 1)
    else:
        data = np.array([xi, yi]).T
        for index, pivot in enumerate(data):
            adjacent_pivots = pivot_adjacent(data.tolist(), index)
            shape, rotation = shape_coordinates(np.array(adjacent_pivots))

            surf = pygame.transform.rotate(texture, rotation)
            screen.blit(surf, pivot + track_offset)


def create_text_renderer(screen):
    font = pygame.font.SysFont('Comic Sans MS', 24)

    def text_render(text, position):
        surface = font.render(text, False, (0, 0, 0))
        screen.blit(surface, position)

    return text_render


def track_render():
    pass


def track_creator():
    # Set full screen centered
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()

    # Set the height and width of the screen
    size = [700, 700]
    screen = pygame.display.set_mode(size)

    # Create a text renderer helper function
    text_renderer = create_text_renderer(screen)

    pygame.display.set_caption("Pivots")

    # Loop until the user clicks the close button.
    done = False
    clock = pygame.time.Clock()

    # Generate pivots
    track_offset = (np.mean(size) * 0.1).astype(int)
    track_size = np.array(size) - track_offset * 2.0

    points, pivots = pivot_map_generate(32, *track_size)
    noise = np.random.random(len(pivots) + 1) * 25.03

    texture = pygame.image.load("./assets/tunnel_road.jpg")
    texture = pygame.transform.scale(texture, (35, 35))

    while not done:

        # This limits the while loop to a max of 10 times per second.
        # Leave this out and we will use all CPU we can.
        clock.tick(10)

        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        # Clear the screen and set the screen background
        screen.fill(WHITE)

        for point in points:
            pygame.draw.circle(screen, BLACK, point + track_offset, 5)

        for index, pivot in enumerate(pivots):
            text_renderer(str(index), pivot + track_offset)

            pygame.draw.circle(screen, RED, pivot + track_offset, 5)

        interpolated_map_generate(screen, texture, np.array(pivots), track_offset, noise = noise)

        pygame.display.flip()

    # Be IDLE friendly
    pygame.quit()
