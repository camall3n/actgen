from collections import namedtuple, defaultdict
import csv
from distutils.util import strtobool
import logging
from pydoc import locate

from matplotlib import pyplot as plt

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


def load_hyperparams(filepath):
    params = dict()
    with open(filepath, newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for name, value, dtype in reader:
            if dtype == 'bool':
                params[name] = bool(strtobool(value))
            else:
                params[name] = locate(dtype)(value)
    return params


def save_hyperparams(filepath, params):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for name, value in sorted(params.items()):
            type_str = defaultdict(lambda: None, {
                bool: 'bool',
                int: 'int',
                str: 'str',
                float: 'float',
            })[type(value)] # yapf: disable
            if type_str is not None:
                writer.writerow((name, value, type_str))


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def update_param(params, name, value):
    if name not in params:
        raise KeyError(
            "Parameter '{}' specified, but not found in hyperparams file.".format(name))
    else:
        logging.info("Updating parameter '{}' to {}".format(name, value))
    params[name] = type(params[name])(value)


def every_n_times(n, count, callback, *args, final_count=None):
    if (count % n == 0) or (final_count is not None and (count == final_count)):
        callback(*args)


def plot_training_gscore(fname='results/training_gscore.csv'):
    """
    plot the +/- g score over time as training proceeds
    """
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        g_scores = list(reader)
        time = [int(i[0]) for i in g_scores]
        plus_g = [float(i[1]) for i in g_scores]
        minus_g = [float(i[2]) for i in g_scores]
        ratio = [plus_g[i] / minus_g[i] for i in range(len(plus_g))]
        diff = [plus_g[i] - minus_g[i] for i in range(len(plus_g))]

        plt.figure()
        plt.plot(time, plus_g, label='+g')
        plt.plot(time, minus_g, label='-g')
        # plt.plot(time, ratio, 'g', label='ratio')
        plt.plot(time, diff, 'r', label='difference')
        plt.title('g score over time during training with SGD')
        plt.xlabel('training step')
        plt.ylabel('g score')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    plot_training_gscore()
