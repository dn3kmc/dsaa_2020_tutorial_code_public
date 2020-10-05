import bayesian_changepoint_detection.online_changepoint_detection as oncd
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import matplotlib.cm as cm

def determine_concept_drift(df):
    R1, maxes = oncd.online_changepoint_detection(df.iloc[:,1], partial(oncd.constant_hazard, 250), oncd.StudentT(0.1, .01, 1, 0))
    # choose 95th percentile for vmax
    sparsity = 5  # only plot every fifth data for faster display
    unflattened_post_probs = -np.log(R1[0:-1:sparsity, 0:-1:sparsity])
    post_probs = (-np.log(R1[0:-1:sparsity, 0:-1:sparsity])).flatten()
    chosen_vmax = int(np.percentile(post_probs, 5))

    plt.pcolor(np.array(range(0, len(R1[:, 0]), sparsity)),
               np.array(range(0, len(R1[:, 0]), sparsity)),
               unflattened_post_probs,
               cmap=cm.gray, vmin=0, vmax=chosen_vmax)
    plt.xlabel("time steps")
    plt.ylabel("run length")
    cbar = plt.colorbar(label="P(run)")
    cbar.set_ticks([0, chosen_vmax])
    cbar.set_ticklabels([1, 0])
    # black = highest prob
    # white = lowest prob
    # the colors mean the same as in arxiv paper
    # the bar direction is just reversed
    plt.show()