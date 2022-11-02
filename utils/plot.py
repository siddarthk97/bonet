from matplotlib import pyplot as plt
import numpy as np 
import torch

from utils.utils import cumargmin

def plot_simple_regret(**kwargs):
    """
    Takes a sequence of pointwise regrets as input
    regrets --> k x n array or n array, k is number of runs
    """

    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    for label, regrets in kwargs.items():
        if len(regrets.shape) == 2:
            simple_regret = np.minimum.accumulate(regrets, axis=1)

            means = np.median(simple_regret, axis=0)
            low = np.percentile(simple_regret, q=0.3, axis=0)
            high = np.percentile(simple_regret, q=0.7, axis=0)
            # means = np.mean(simple_regret, axis=0)
            # low = np.std(simple_regret, axis=0)
            # high = np.std(simple_regret, axis=0)
            '''
            # means = np.mean(regrets, axis=0)
            # std = np.std(regrets, axis=0)
            m = np.median(regrets, axis=0)
            low = np.percentile(regrets, q=0.3, axis=0)
            high = np.percentile(regrets, q=0.7, axis=0)

            means = np.minimum.accumulate(m, axis=0)
            idx = cumargmin(m)
            '''

            line = ax.plot(range(len(means)), means, '-', label=label)
            ax.fill_between(range(len(means)), means-low, means+high, alpha=0.3, facecolor=line[-1].get_color())
        else:
            simple_regret = np.minimum.accumulate(regrets)
            ax.plot(range(len(simple_regret)), simple_regret, '-', label=label)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    # plt.ylim([0, 50])

    plt.xlabel('Evaluations')
    plt.ylabel('Simple Regret')

def plot_pointwise_regret(x, simple_regrets):
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    ax.plot(x, simple_regrets, '-')
    ax.plot(x, x, '-')


if __name__ == "__main__":
    results = np.arange(20)[::-1]
    results = np.asarray([results] * 5)
    noise = np.random.normal(scale=1, size=results.shape)
    results = results + noise

    means = np.mean(results, axis=0)
    std = np.std(results, axis=0)
