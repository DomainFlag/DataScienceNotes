import numpy as np
import matplotlib.pyplot as plt


def show_frame(frame, title = 'Frame'):
    plt.figure()
    plt.imshow(frame.permute(1, 2, 0).squeeze(2).numpy(), interpolation = 'none', cmap = 'gray')
    plt.title(title)
    plt.show()


def show_features(features, title, workers = 1, step = 20, save = False, version = 1.0):
    """ If workers > 1, features must be a time series of results based on latest model optimization """
    if len(features) < step:
        return

    data = np.array(features)

    # Model optimization step domain
    if workers > 1:
        data = data.reshape(-1, workers).mean(axis = 1)

    # Pad the data based on step size
    data = np.concatenate((data, data[-(len(data) % step):])).reshape(-1, step)

    # Features extraction
    x = np.arange(0, len(data)) * step
    y, sd = data.mean(axis = 1), data.std(axis = 1)

    plt.plot(x, y, linewidth = 2)
    plt.fill_between(x, (y - sd / 2), (y + sd / 2), color = 'b', alpha = .05)
    plt.title(title)

    if save:
        plt.savefig(f"./static/{title}_{version}.png")

    plt.show()


def smoothness(x):
    """ X value is expected to be normalized - [0, 1] """
    assert 0.0 <= x <= 1.0, "x < 0 or x > 1.0, x - %s" % (x,)

    return (np.log(x + 1.0) / np.log(2)) ** (1 / 2)


def compute_min_offset(a, b, cycle):
    offset = a - b

    if abs(offset) < cycle - abs(offset):
        return offset
    else:
        return cycle * (-1 if a > b else 1) + offset
