import attention as atn
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

def alignmentTable(query, definition, a_tilde, b_tilde):
    """
    Given query, definition, a_tilde, b_tilde from attention.py,
    returns heatmap of the dot product
    """
    nrows, ncols = len(query), len(definition)
    image = np.zeros((nrows, ncols)) 

    for i in range(nrows):
        for j in range(ncols):
            image[i,j] = np.dot(a_tilde[i], b_tilde[j])

    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False)
    cax = ax.matshow(image, cmap='gray')

    ax.set_xticklabels(['']+query)
    ax.set_yticklabels(['']+definition)

    plt.tick_params(labelsize=8)
    plt.show()
