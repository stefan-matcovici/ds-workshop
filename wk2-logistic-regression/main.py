import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import toimage


def read_x(filename):
    with open(filename) as f:
        all_text = f.read().split('\n')[:-1]
        lines = np.array(list(map(lambda x: list(map(int, x.split(' '))), all_text)))
    return lines


def read_y(filename):
    with open(filename) as f:
        all_text = f.read().split('\n')[:-1]
        lines = np.array(list(map(int, all_text)))
    return lines


def show_imgs(X):
    plt.figure(1)
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            plt.subplot2grid((4, 4), (i, j))
            plt.imshow(X[k].reshape(8, 8))
            k = k + 1
    # show the plot
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_error(w, x, y, l):
    return l / 2 * np.sum(w ** 2) - np.sum(y * np.log(sigmoid(x.dot(w))) + (1 - y) * np.log(1 - sigmoid(x.dot(w))))


def add_ones_column(x, shape):
    temp_x = np.ones(shape)
    temp_x[:, :-1] = x

    return temp_x


if __name__ == "__main__":
    x_train = read_x("./data/digit_x.dat")
    y_train = read_y("./data/digit_y.dat")

    x_train = add_ones_column(x_train, (100, 65))

    x_test = read_x("./data/digit_x_test.dat")
    y_test = read_y("./data/digit_y_test.dat")

    x_test = add_ones_column(x_test, (400, 65))

    weights = np.random.randn(x_train.shape[1], 1)
    y_train = y_train.reshape(-1, 1)

    no_iterations = 10000
    learning_rate = 0.1 / 80
    regularization = 1
    train_accuracy = []
    test_accuracy = []
    while no_iterations > 0:
        updates = regularization * weights - x_train.T.dot(y_train - sigmoid(x_train.dot(weights)))
        weights = weights - learning_rate * updates

        predictions = np.array([1 if x > 0 else 0 for x in x_train.dot(weights)]).reshape((-1, 1))
        error = np.sum(abs(y_train - predictions))
        train_accuracy.append((100 - error) / 100)

        predictions = [1 if x > 0 else 0 for x in x_test.dot(weights)]
        error = np.sum(abs(y_test - predictions))
        test_accuracy.append((400 - error) / 400)

        no_iterations -= 1

    fig, ax = plt.subplots()
    ax.plot(train_accuracy, color='C2', label='train accuracy')
    ax.plot(test_accuracy, color='C1', label='test accuracy')
    ax.legend()
    plt.show()

    predictions = [1 if x > 0 else 0 for x in x_test.dot(weights)]
    error = np.sum(abs(y_test - predictions))
    print((400 - error) / 400)
