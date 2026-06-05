from parameters import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Wing():
    def __init__(self, geometry, forces, alpha=0.0):
        self.geometry = geometry
        self.forces = forces
        self.alpha = alpha
        self.weight = ([0, 0.5, 0], [0, 0, 0.5])
        self.axes = np.eye(3)

    def rotate_around_axis(self, axis, angle):
        self.geometry = rotate_vectors_around_axis(axis, angle, self.geometry)

        self.axes = rotate_vectors_around_axis(axis, angle, self.axes)

        for i in range(len(self.forces)):
            print(self.forces[i][0])
            print(self.forces[i][1])
            total_vector = self.forces[i][0] + np.array(self.forces[i][1])
            total_vector = rotate_vectors_around_axis(axis, angle, total_vector)
            rotated_origin = rotate_vectors_around_axis(axis, angle, self.forces[i][1])
            self.forces[i] = (total_vector - rotated_origin, rotated_origin)

            #full_vector = self.forces[i][0] + np.array(self.forces[i][1])
            #full_vector = rotate_vectors_around_axis(axis, angle, full_vector)
            #origin_vector = rotate_vectors_around_axis(axis, angle, np.array(self.forces[i][1]))

        self.weight = (self.weight[0], rotate_vectors_around_axis(axis, angle, self.weight[1]))

    def plot_wing(self, alpha=None, plotwing=True, plotforces=True, plotaxes=True):
        if alpha != None:
            self.alpha = alpha
        if plotwing:
            plot_quadrilateral(ax, self.geometry, face_color='cyan', edge_color='black', alpha=self.alpha)
        if plotforces:
            for force in self.forces:
                plot_single_vector(ax, force[0], origin=force[1], color="purple")
            plot_single_vector(ax, self.weight[0], origin=self.weight[1], color="orange")
        if plotaxes:
            plot_axes(self.axes)


def transform_axes(theta_deg, phi_deg):
    """
    Transforms the three standard unit axes by rotating theta around the Y axis,
    and then -phi around the Z axis.

    Parameters:
    theta_deg (float): Rotation angle around Y axis in degrees.
    phi_deg (float): Rotation angle around Z axis in degrees (will be negated inside).

    Returns:
    transformed_axes (np.ndarray): A 3x3 matrix where each column represents
                                   the new X, Y, and Z unit vectors.
    """
    # Convert angles from degrees to radians
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)  # Keep positive here, handled in the matrix

    # 1. Rotation matrix around Y axis by theta
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    # 2. Rotation matrix around Z axis by -phi
    # Transformation done around -phi, not phi, so sines are flipped
    R_z = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [+np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix: R = R_z(-phi) @ R_y(theta)
    # This applies R_y first, then R_z
    R_combined = np.dot(R_z, R_y)

    # Standard unit axes defined as columns of an identity matrix
    # Columns: [X, Y, Z]
    unit_axes = np.eye(3)

    # Transform the axes
    transformed_axes = np.dot(R_combined, unit_axes)

    return transformed_axes
def decompose(vec, axes):
    return np.array([project_vector(vec, axes[0]), project_vector(vec, axes[1]), project_vector(vec, axes[2])])
def rotate_vectors_around_axis(d, alpha_deg, vectors=None):
    """
    Rotates a set of 3D vectors around an arbitrary axis vector d by angle alpha.

    Parameters:
    d (array-like): The 3D vector defining the axis of rotation.
    alpha_deg (float): The rotation angle in degrees.
    vectors (np.ndarray, optional): An (N, 3) array of vectors to rotate.
                                    Defaults to the standard 3x3 unit axes.

    Returns:
    rotated_vectors (np.ndarray): The transformed vectors with the same shape as input.
    """
    # 1. Normalize the rotation axis vector d
    d = np.asarray(d, dtype=float)
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        raise ValueError("The rotation axis vector 'd' cannot be a zero vector.")
    u = d / d_norm  # Unit axis

    # 2. If no vectors are specified, default to the 3x3 standard unit axes
    if vectors is None:
        vectors = np.eye(3)
    else:
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1 and len(vectors) == 3:
            # Handle edge case where a single vector [x, y, z] is passed instead of [[x, y, z]]
            vectors = vectors.reshape(1, 3)
        elif vectors.shape[-1] != 3:
            raise ValueError("The 'vectors' array must have a shape of (N, 3).")

    # 3. Convert angle to radians
    alpha = np.radians(alpha_deg)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    # 4. Construct the skew-symmetric cross-product matrix K
    K = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])

    # 5. Compute Rodrigues' Rotation Matrix (R)
    I = np.eye(3)
    R = I + sin_a * K + (1 - cos_a) * np.dot(K, K)

    # 6. Transform the vectors.
    # Since 'vectors' is shape (N, 3), we transpose it to (3, N) to multiply with R (3, 3),
    # then transpose the result back to (N, 3).
    rotated_vectors = np.dot(R, vectors.T).T

    # If the user passed a single 1D vector, squeeze it back to a 1D vector for convenience
    if rotated_vectors.shape[0] == 1:
        rotated_vectors = rotated_vectors.squeeze()

    return rotated_vectors
def rotate_axes_around_vector(d, alpha_deg):
    """
    Rotates the standard unit axes around an arbitrary vector d by angle alpha.

    Parameters:
    d (array-like): The 3D vector defining the axis of rotation.
    alpha_deg (float): The rotation angle in degrees.

    Returns:
    rotated_axes (np.ndarray): A 3x3 matrix where columns are the new X, Y, Z axes.
    """
    # 1. Normalize the direction vector d to ensure it's a unit vector
    d = np.asarray(d, dtype=float)
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        raise ValueError("The rotation axis vector 'd' cannot be a zero vector.")
    u = d / d_norm  # u is now our unit axis [ux, uy, uz]

    # 2. Convert angle to radians
    alpha = np.radians(alpha_deg)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    # 3. Construct the cross-product symmetric matrix (K) of the unit vector u
    # This represents the operation (u x v) as a matrix multiplication K @ v
    K = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])

    # 4. Rodrigues' Rotation Formula to get the 3x3 rotation matrix R
    # R = I + sin(alpha)*K + (1 - cos(alpha))*K^2
    I = np.eye(3)
    R = I + sin_a * K + (1 - cos_a) * np.dot(K, K)

    # 5. Rotate the standard unit axes (represented as the Identity matrix)
    # R @ I is just R, where columns of R are the newly transformed axes
    rotated_axes = R

    return rotated_axes
def plot_single_vector(ax, vector, origin=(0, 0, 0), color='b', label=None):
    """
    Plots a single 3D vector on a given Matplotlib axis.
    Safely handles both Python lists and numpy arrays (including column vectors).
    """
    # 1. Force inputs into flat (1D) numpy arrays of shape (3,)
    vector = np.asarray(vector).flatten()
    origin = np.asarray(origin).flatten()

    # Optional dimension check to catch errors early
    if len(vector) != 3 or len(origin) != 3:
        raise ValueError("Both vector and origin must have exactly 3 components.")

    # 2. Unpack safely
    x_start, y_start, z_start = origin
    u, v, w = vector

    # 3. Plot
    ax.quiver(x_start, y_start, z_start,
              u, v, w,
              color=color, arrow_length_ratio=0.1, linewidth=2, label=label)
def plot_quadrilateral(ax, vertices, scale_factor=0.1, face_color='cyan', edge_color='black', alpha=0.4):
    """
    Plots a 3D quadrilateral (or any polygon) from a list of vertices.

    Parameters:
    ax (Axes3D): The matplotlib 3D axis object.
    vertices (list or array): A list of (x, y, z) tuples/arrays defining the corners.
    scale_factor (float): Multiplier to scale the size of the quadrilateral.
    face_color (str): Color of the solid polygon.
    edge_color (str): Color of the boundary lines.
    alpha (float): Transparency (0.0 is invisible, 1.0 is solid).
    """
    # Ensure vertices are standard numpy arrays and apply the scale factor
    verts = np.asarray(vertices) * scale_factor

    # Poly3DCollection requires a list of polygons.
    # Since we are drawing one polygon, we wrap our vertices in a single outer list.
    polygon = Poly3DCollection([verts], alpha=alpha, facecolors=face_color, edgecolors=edge_color)

    # Add the polygon to the axis
    ax.add_collection3d(polygon)
def plot_axes(axis):
    plot_single_vector(ax, axis[0], color='r', label='Vector 1')
    plot_single_vector(ax, axis[1], color='g', label='Vector 1')
    plot_single_vector(ax, axis[2], color='b', label='Vector 1')


if __name__ == "__main__":

    #Initialize plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')


    ###3 coords systems used in this model, fuselage (a.c), wing and hinge

    theta_input = -45 #deg
    phi_input = 36 #deg
    hinge_axes = transform_axes(theta_input, phi_input)
    hinge_vector = hinge_axes[0]


    wing_planform = Wing(
                        np.array([[2, 0, 0], [1, 0, 10], [-1, 0, 10], [-1, 0, 0]]),
                [
                    (np.array([0.3, 0, 0]), np.array([0, 0, 0.3])),
                    (np.array([0.3, 0, 0]), (0, 0, 0.7)),
                       ])


    steps = 50
    for i in range(steps + 1):
        wing_planform.plot_wing(0.4/(steps + 1)*i, plotwing=False, plotaxes=False)
        wing_planform.rotate_around_axis(hinge_vector, 120/steps)

    wing_axes = rotate_vectors_around_axis(hinge_vector, 0)

    #plot_single_vector(ax, np.array([0.3, 0, 0]), origin=(0, 0, 0.7), color="purple")
    #plot_single_vector(ax, np.array([0.3, 0, 0]), origin=(0, 0, 0.3), color="purple")


    plot_axes(np.eye(3))
    plot_axes(wing_axes)
    plot_single_vector(ax, hinge_vector, color='y', label='Vector 3')

    # --- Critical Step: Manually set limits for 3D quiver plots ---
    # Find the maximum absolute value among all vectors to make a neat cube
    all_vectors = np.array([wing_axes[0], wing_axes[1], wing_axes[2]])
    max_val = np.max(np.abs(all_vectors))

    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # 4. Add finishing touches
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.view_init(elev=15, azim=45, vertical_axis='y')
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.legend()

    plt.show()