# Script for plotting train/validate performance for a batch of runs
import os
import sys

from matplotlib import pyplot as plt
from parse import parse

from Hyd_ML import moving_average
from Util import *

def plot_one(sd):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(sd.name)
    ax.set_ylabel("train/val loss")
    ax.set_xlabel("epoch")

    dirs = [d for d in os.scandir(sd) if d.is_dir()]
    dirs.sort(key=lambda x: parse_output_num(x))
    labels = [parse_output_num(d) * 2 for d in dirs]
    vals = []
    colors = plt.cm.jet(np.linspace(0, 1, len(labels)))
    for dir, label, col in zip(dirs, labels, colors):
        path = os.path.join(dir, "models/progress.log")
        progress = open(path, 'r')
        lines = progress.readlines()

        train = []
        val = []
        for line in lines:
            res = parse('Median validation NSE epoch {}/{} = {} training NSE {}', line)
            if res is not None:
                train.append(float(res[3]))
                val.append(float(res[2]))

        if len(train) < 2:
            print(f"Error: {path} did not run or log did not parse")
        else:
            ax.plot(moving_average(train, i=10), '--', color=col)
            ax.plot(moving_average(val, i=10), color=col, label=str(label))

            vals.append(np.max(moving_average(val, i=10)))  # np.mean(val[-50:])
    ax.set_ylim(bottom=0.65)
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title("Final validation error: " + sd.name)
    ax2.set_ylabel("val loss")
    ax2.set_xlabel("Parameter value")
    ax2.plot(vals, marker="o", linestyle="None")
    ax2.set_xticks(range(len(dirs)))
    ax2.axes.xaxis.set_ticklabels(labels)
    plt.show()


def parse_output_num(x):
    return int(parse('output{}', x.name)[0])


if __name__ == '__main__':
    path = sys.argv[1]

    if os.path.exists(path + "/output1"):
        plot_one(os.path(path))
    else:
        superdirs = [d for d in os.scandir(path) if d.is_dir()]
        for sd in superdirs:
            plot_one(sd)
