import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def f(x, a, b):
    return a * np.sin(b * x)


def get_derivative_with_respect_to_a(x, a, b):
    return np.sin(b * x)


def get_derivative_with_respect_to_b(x, a, b):
    return a * x * np.cos(b * x)


def get_error_derivative(x, y, a, b):
    return y - a * np.sin(b * x)


def get_a_adjustements(x, y, a, b):
    return get_error_derivative(x, y, a, b) * get_derivative_with_respect_to_a(x, a, b)


def get_b_adjustements(x, y, a, b):
    return get_error_derivative(x, y, a, b) * get_derivative_with_respect_to_b(x, a, b)


def get_error(x, y, a, b):
    return np.sum((y - a * np.sin(b * x)) ** 2)


def plot_points_with_function_estimation(a, b, x, y, iteration):
    fig, ax = plt.subplots()
    plt.title("Iteration: %d, a=%f, b=%f" % (iteration, a, b))
    ax.scatter(x, y, color='C1', label='real data')
    ax.plot(x, a * np.sin(b * x), color='C2', label='estimator')
    ax.legend()
    plt.show()


def show_error_plot(x, y):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    a = np.arange(-2, 2, 4 / 100)
    b = np.arange(-2, 2, 4 / 100)
    a, b = np.meshgrid(a, b)

    duplicated_x = np.repeat(x, 100 * 100).reshape(100, 100, 100)
    duplicated_y = np.repeat(y, 100 * 100).reshape(100, 100, 100)

    z = np.sum((duplicated_y - a * np.sin(b * duplicated_x)) ** 2, axis=2)

    # Plot the surface.
    surf = ax.plot_surface(a, b, z, cmap=cm.coolwarm,
                          linewidth=0, antialiased=False, alpha=0.5)

    # Customize the z axis.
    ax.view_init(50, 75)
    # ax.set_zlim(0, 80)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__ == "__main__":
    train = np.genfromtxt('./data/data.txt', delimiter=',', skip_header=1)
    parameters = np.zeros((train.shape[1],))
    x = np.array([t[0] for t in train])
    y = np.array([t[1] for t in train])

    no_iterations = 50
    learning_rate = 0.001
    a = np.random.rand()
    b = np.random.rand()

    errors = []
    a_s = []
    b_s = []
    i = 0
    while i < no_iterations:
        i += 1
        error = get_error(x, y, a, b)
        errors.append(error)

        plot_points_with_function_estimation(a, b, x, y, i)
        plt.pause(0.1)

        a_s.append(a)
        b_s.append(b)
        a += learning_rate * 2 * np.sum(get_a_adjustements(x, y, a, b))
        b += learning_rate * 2 * np.sum(get_b_adjustements(x, y, a, b))

    fig, ax = plt.subplots()
    ax.plot(errors, color='C2', label='error')
    ax.legend()
    plt.show()

    # plot_points_with_function_estimation(a, b, x, y)
    show_error_plot(x, y)
