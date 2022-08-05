import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 160
import os
from os.path import join
import argparse
from matplotlib import lines

#mpl.rc('text', usetex=True)


def parseArguments():
    ag = argparse.ArgumentParser()
    ag.add_argument('-b', '--base_path', help='Specify base_path to data directory', required=False, default='./test_cases')
    ag.add_argument('-d', '--data_directory', help='Specify data directory', required=False, default='.')
    args = ag.parse_args()
    return args


def main(args):
    # setting
    base_path = args.base_path
    directory = args.data_directory
    data_path = join(base_path, directory)

    filenames= os.listdir(data_path)
    cases = []
    for filename in filenames: 
        if os.path.isdir(os.path.join(os.path.abspath(data_path), filename)):
            cases.append(filename)

    data = {}

    lineStyle = list(lines.lineStyles.keys())[0:4]
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, case in enumerate(cases):
        case_path = join(data_path, case)
        case_data_file = join(case_path, "postProcessing/forces/0/coefficient.dat")
        data[case] = np.loadtxt(case_data_file  , unpack=True, usecols=[0, 1, 3])
        ax.plot(data[case][0], data[case][1], lineStyle[idx%len(lineStyle)], label=case)

    ax.set_xlim((0, 6))
    ax.set_ylim((2.5, 3.5))
    ax.set_ylabel(r"$c_D$", fontsize=12)
    ax.set_xlabel(r"$\tilde t$", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.legend(loc='best', fontsize=12)

    plt.savefig(join(data_path, 'cd.png'))

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, case in enumerate(cases):
        case_path = join(data_path, case)
        case_data_file = join(case_path, "postProcessing/forces/0/coefficient.dat")
        data[case] = np.loadtxt(case_data_file  , unpack=True, usecols=[0, 1, 3])
        ax.plot(data[case][0], data[case][2], lineStyle[idx%len(lineStyle)], label=case)

    ax.set_xlim((0, 6))
    ax.set_ylim((-1.6, 1.6))
    ax.set_ylabel(r"$c_L$", fontsize=12)
    ax.set_xlabel(r"$\tilde t$", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.legend(loc='best', fontsize=12)

    plt.savefig(join(data_path, 'cl.png'))

if __name__ == "__main__":
    args = parseArguments()
    main(args)
