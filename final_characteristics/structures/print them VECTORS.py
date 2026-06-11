import numpy as np
import matplotlib.pyplot as plt

def plot_vectors_3d(vectors):
    vectors = np.asarray(vectors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Draw arrows from origin
    origins = np.zeros_like(vectors)

    ax.quiver(
        origins[:, 0],
        origins[:, 1],
        origins[:, 2],
        vectors[:, 0],
        vectors[:, 1],
        vectors[:, 2],
        arrow_length_ratio=0.08,
        normalize=False,   # preserve actual vector magnitudes
    )

    # Equal scaling for x/y/z axes
    max_range = np.abs(vectors).max()

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # Force equal aspect ratio (important!)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

vectors = np.loadtxt("r_moments.txt")
plot_vectors_3d(vectors)