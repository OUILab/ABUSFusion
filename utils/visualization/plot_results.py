# utils/visualization/plot_results.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_3d_volume(positions):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = positions[:, 4], positions[:, 5], positions[:, 6]
    ax.scatter(x, y, z, c=range(len(x)), cmap="viridis")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reconstructed 3D Ultrasound Volume")

    plt.show()
