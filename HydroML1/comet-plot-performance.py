# Script for plotting train/validate performance for a batch of runs
import os
import sys

from matplotlib import pyplot as plt
from parse import parse

from Hyd_ML import moving_average, median_filter, savefig, save_show_close
from Util import *


def plot_one(title, dirlist, transform=lambda x: x, parameter_name="Parameter value", all_train_val=True, subtitle=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    titles = False
    if titles:
        fig.suptitle(title)
        ax.set_title(title)
        if subtitle is not None:
            ax.set_title(subtitle)
    ax.set_ylabel("Training/Validation NSE")
    ax.set_xlabel("Epoch")

    dirs = []
    for onedir in dirlist:
        dirs += [d for d in os.scandir(onedir) if d.is_dir() and parse('output{}', d.name) is not None]

    dirs.sort(key=lambda x: parse_output_num(x))
    labels = [transform(parse_output_num(d)) for d in dirs]
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
            mav_len = 100 if all_train_val else None
            mf_len = 9
            if all_train_val:
                ax.plot(median_filter(train, mf_len), '--', color=col)
            else:
                ax.plot(median_filter(train, mf_len), color='g',
                        label=None if all_train_val else "Training NSE")
            ax.plot(median_filter(val, mf_len), color=col, label=str(label) if all_train_val else "Validation NSE")

            vals.append(np.max(moving_average(val, i=10)))  # np.mean(val[-50:])
        if not all_train_val:
            break

    if all_train_val:
        pass
        #ax.set_ylim(bottom=0.65)
    else:
        ax.set_ylim(bottom=0.5)

    ax.legend()
    plt.rcParams.update({'font.size': 16})
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    fig.tight_layout()
    save_show_close('TrainValidate-' + title, plt, fig)

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    if titles:
        ax2.set_title("Maximum validation NSE vs. " + title)
    ax2.set_ylabel("Maximum validation NSE")
    ax2.set_xlabel(parameter_name)
    #if len(vals) > 1:
    #    ax2.plot(labels, vals, marker="o", linestyle="None")
    #else:
    ax2.plot(vals, marker="o", linestyle="None")
    ax2.set_xticks(range(len(labels)))
    ax2.axes.xaxis.set_ticklabels(labels)
    #current_values = ax2.axes.xaxis.get_ticklabels()
    #ax2.axes.xaxis.set_ticklabels(['{}'.format(int(str(x))) for x in np.unique(labels)])
    #loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
    #ax.xaxis.set_major_locator(loc)
    #ax.xaxis.set_minor_locator(loc)
    plt.rcParams.update({'font.size': 16})
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    fig.tight_layout()
    save_show_close(title, plt, fig)


def parse_output_num(x):
    return int(parse('output{}', x.name)[0])


if __name__ == '__main__':
    if True:
        path = sys.argv[1]

        if os.path.exists(path + "/output1"):
            d = os.path(path)
            plot_one(d.name, [d])
        else:
            superdirs = [d for d in os.scandir(path) if d.is_dir()]
            for sd in superdirs:
                plot_one(sd.name, [sd])
    else:
        # Make paper figs:
        # 2 hasn't run to convergence yet
        #plot_one("Encoding length", [r"C:\hydro\from_comet\stores-sigs-runs3\run-encodinglength", r"C:\hydro\from_comet\stores-sigs-runs3\run-encodinglength2"],
        #         parameter_name="Years per sample")
        plot_one("Encoding length", [r"C:\hydro\from_comet\stores-sigs-runs3\run-encodinglength2"],
                 parameter_name="Encoding length (years per sample)")
        plot_one("Median training and validation error", [r"C:\hydro\from_comet\stores-sigs-runs3\run-encodinglength"],
                 all_train_val=False, subtitle="Initial median NSE = -0.259")

        plot_one("Number of signatures", [r"C:\hydro\from_comet\stores-sigs-runs3\run-numsigs"],
                 parameter_name="Number of learnt signatures", transform=lambda x: 2*x)
        plot_one("Number of stores", [r"C:\hydro\from_comet\stores-sigs-runs3\run-numstores"],
                 parameter_name="Number of stores", transform=lambda x: 2*x)
