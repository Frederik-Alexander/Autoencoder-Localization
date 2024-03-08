"""Functions for visualizing stuff."""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools

def visualize_latents(latents, labels, save_file=None):
    try:
        latents = latents.cpu()
    except:
        pass
    # Assuming latents is an n-dimensional array
    num_dims = latents.shape[1]
    if int(num_dims) == 2:
        visualize_latents_OLD(latents, labels, save_file)
        return
    if int(num_dims) == 3:
        visualize_latents_Three(latents, labels, save_file)
        return
    # Number of subplot rows and columns
    num_plots = num_dims * (num_dims - 1) // 2
    num_rows = int(num_plots**0.5)
    num_cols = (num_plots + num_rows - 1) // num_rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))  # Adjust figsize as needed

    plot_number = 0
    for i, j in itertools.combinations(range(num_dims), 2):
        row = plot_number // num_cols
        col = plot_number % num_cols
        ax = axs[row, col]

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'AE - Dimensions {i+1} & {j+1}')
        ax.scatter(latents[:, i], latents[:, j], cmap=plt.cm.coolwarm, s=3., alpha=0.5) #img

        plot_number += 1

    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=600)
    plt.close()
    #plt.show()

def visualize_latents_Three(latents, labels, save_file,num_dims=3):

        num_plots = num_dims * (num_dims - 1) // 2
        num_rows = int(np.ceil(np.sqrt(num_plots)))
        num_cols = int(np.ceil(num_plots / num_rows))

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))  # Adjust figsize as needed
        fig.tight_layout(pad=4.0)

        plot_number = 0
        for i, j in itertools.combinations(range(num_dims), 2):
            row = plot_number // num_cols
            col = plot_number % num_cols
            ax = axs[row, col] if num_rows > 1 else axs[col]

            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect('equal')
            ax.set_title(f'AE - Dimensions {i+1} & {j+1}')
            ax.scatter(latents[:, i], latents[:, j], cmap=plt.cm.coolwarm, s=5., alpha=1)

            plot_number += 1

        plt.tight_layout()
        if save_file:
            plt.savefig(save_file, dpi=600)
        plt.close()

def visualize_latents_OLD(latents, labels, save_file=None):
    fig, ax = plt.subplots()
    ax.set_xlim(xmin=-0.7, xmax=0.7)
    ax.set_ylim(ymin=-0.7, ymax=0.7)
    ax.set_aspect('equal')
    ax.set_title('AE')
    ax.scatter(latents[:, 0], latents[:, 1], # , c=labels
                cmap=plt.cm.coolwarm, s=2., alpha=0.5)

    if "Extra" in save_file:
        print(latents)
        plt.savefig(save_file, dpi=200)
        #plt.show()
        plt.close()

def plot_losses(losses, losses_std=defaultdict(lambda: None), save_file=None):
    """Plot a dictionary with per epoch losses.

    Args:
        losses: Mean of loss per epoch
        losses_std: stddev of loss per epoch

    """
    plt.close()
    for key, values in losses.items():
        plt.errorbar(range(len(values)), values, yerr=losses_std[key], label=key)

    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    if save_file:
        plt.savefig(save_file, dpi=200)
        plt.close()


def shape_is_image(shape):
    """Check if is a 4D tensor which we consider to be an image."""
    return len(shape) == 4

