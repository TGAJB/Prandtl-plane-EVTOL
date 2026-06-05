from parameters import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import csv

class Wing():
    def __init__(self, geometry, theta_deg, phi_deg, alpha=0.0):
        #plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.geometry = geometry
        self.point_loads = []
        self.nonangled_point_loads = [PointLoad((0, 200, 0), (0, 0, 5), color="orange")]
        self.moments = []
        self.distributed_loads = []
        self._angle = 0
        self.hinge_axes = transform_axes(theta_deg, phi_deg)
        self.hinge_vector = self.hinge_axes[0]
    @property
    def angle(self):
        return self._angle
    @angle.setter
    def angle(self, value):

        for point_load in self.nonangled_point_loads:
            point_load.dir = rotate_vectors_around_axis(self.hinge_vector, self._angle, point_load.dir)

        self._angle = value

        for point_load in self.nonangled_point_loads:
            point_load.dir = rotate_vectors_around_axis(self.hinge_vector, -self._angle, point_load.dir)
    def add_point_load(self, load):
        if isinstance(load, PointLoad):
            self.point_loads.append(load)
        else: raise TypeError
    def add_distributed_load(self, distributed_load):
        if isinstance(distributed_load, DistributedLoad):
            self.distributed_loads.append(distributed_load)
        else: raise TypeError
    def plot_wing(self):

        plot_quadrilateral(self.ax, rotate_vectors_around_axis(self.hinge_vector, self.angle, self.geometry), alpha=0)

        plot_single_vector(self.ax, self.hinge_vector, color="yellow")

        #find max scale factor among loads
        maxload = 0
        for load in self.point_loads:
            if load.magn > maxload:
                maxload = load.magn
        for load in self.nonangled_point_loads:
            if load.magn > maxload:
                maxload = load.magn

        maxload /= 1

        for point_load in self.point_loads:
            point_dir_rotated = rotate_vectors_around_axis(self.hinge_vector, self.angle, point_load.dir)
            point_loc_rotated = rotate_vectors_around_axis(self.hinge_vector, self.angle, point_load.loc)
            plot_single_vector(self.ax, point_dir_rotated*point_load.magn/maxload, point_loc_rotated, point_load.color)

        for point_load in self.nonangled_point_loads:
            point_dir_rotated = rotate_vectors_around_axis(self.hinge_vector, self.angle, point_load.dir)
            point_loc_rotated = rotate_vectors_around_axis(self.hinge_vector, self.angle, point_load.loc)
            plot_single_vector(self.ax, point_dir_rotated*point_load.magn/maxload, point_loc_rotated, point_load.color)


        number_of_points = 10 #points for plotting distributed loads

        for load in self.distributed_loads:

            geo = []
            geo.append(load.loc)
            for i in range(number_of_points + 1):
                x = np.sqrt(np.dot(load.load_dir, load.load_dir))/number_of_points*i
                y = load.func(x)
                load_dir_normalized = load.load_dir/np.sqrt(np.dot(load.load_dir, load.load_dir))
                geo.append(load.loc + load.dir*y + load_dir_normalized*x)


            geo.append(load.loc + load.load_dir)

            plot_polygon_3d(self.ax, rotate_vectors_around_axis(self.hinge_vector, self.angle, geo), face_color=load.color)

            #geo = (load.loc, load.loc + load.dir*load.magn/maxload, load.loc + load.dir*load.magn/maxload + load.load_dir, load.loc + load.load_dir)

            #plot_quadrilateral(self.ax, rotate_vectors_around_axis(self.hinge_vector, self.angle, geo), face_color=load.color)


        plot_axes(self.ax)

        #plot_axes(self.ax, self.hinge_axes)

        # 4. Add finishing touches
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        self.ax.view_init(elev=30, azim=45, vertical_axis='y')
        self.ax.invert_yaxis()
        self.ax.invert_zaxis()
        self.ax.legend()

        # 3. EQUAL SCALING FIX: Calculate the cubic bounding box
        # Force the axes to have equal visual scale
        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        x_mid = np.mean(x_limits)
        y_mid = np.mean(y_limits)
        z_mid = np.mean(z_limits)

        # Find the maximum range to make a cube
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        self.ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
        self.ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
        self.ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

        self.ax.invert_yaxis()
        self.ax.invert_zaxis()

        # Final aspect lock
        try:
            self.ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            # Fallback for older matplotlib versions
            self.ax.set_aspect('equal')

    def discretize(self, res, distributed_mesh_size=11):
        ###Numpy array with discretized force positions

        discretized_forces = []
        discretized_positions = []

        for load in self.point_loads:
            discretized_positions.append(np.round(load.loc / res) * res)
            discretized_forces.append(load.dir*load.magn)

        for load in self.nonangled_point_loads:
            discretized_positions.append(np.round(load.loc / res) * res)
            discretized_forces.append(load.dir*load.magn)

        for load in self.distributed_loads:

            load_dir_magn = np.sqrt(np.dot(load.load_dir,load.load_dir))
            load_dir_norm = load.load_dir/load_dir_magn
            for i in range(distributed_mesh_size):
                x = load_dir_magn/(distributed_mesh_size - 1)*i
                y = load.func(x)
                discretized_forces.append(load.dir * y)
                discretized_positions.append(load.loc + load_dir_norm*x)
            discretized_pos = np.round(load.loc / res) * res


        print("forces")
        print(discretized_forces)
        print("positions")
        print(discretized_positions)

        return np.array(discretized_forces), np.array(discretized_positions)


class Load():
    def __init__(self, force, loc, label="", color="purple"):
        self.dir = np.array(force/np.sqrt(np.dot(force, force)))
        self.loc = np.array(loc)
        #For plotting
        self.label = label
        self.color = color
class PointLoad(Load):
    def __init__(self, force, loc, label="", color="purple"):
        # Inherits everything directly from the parent Load class
        super().__init__(force, loc, label, color)
        self.magn = np.sqrt(np.dot(force, force))
class DistributedLoad(Load):
    def __init__(self, force, loc, load_dir, func, label="", color="purple"):
        super().__init__(force, loc, label, color)
        self.load_dir = np.array(load_dir)
        self.func = func
        self.magn = 5
    @classmethod
    def import_from_csv(cls, file_path):
        pass
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
              color=color, arrow_length_ratio=0.5, linewidth=2, label=label)
def plot_quadrilateral(ax, vertices, scale_factor=1, face_color='cyan', edge_color='black', alpha=0.4):
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
def plot_axes(ax, axis=np.eye(3)):
    plot_single_vector(ax, axis[0], color='r', label='Vector 1')
    plot_single_vector(ax, axis[1], color='g', label='Vector 1')
    plot_single_vector(ax, axis[2], color='b', label='Vector 1')
def plot_polygon_3d(ax, vertices, scale_factor=1, face_color='cyan', edge_color='black', alpha=0.4):
    """
    Plots a 3D polygon from a list of vertices with any number of corners.

    Parameters:
    ax (Axes3D): The matplotlib 3D axis object.
    vertices (list or array): A list of (N, 3) coordinates defining the corners.
                              Must contain at least 3 vertices.
    scale_factor (float): Multiplier to scale the size of the polygon.
    face_color (str): Color of the solid polygon.
    edge_color (str): Color of the boundary lines.
    alpha (float): Transparency (0.0 is invisible, 1.0 is solid).
    """
    # Convert to a numpy array for easy mathematical operations
    verts = np.asarray(vertices)

    # Safety check: A polygon needs at least 3 points (a triangle)
    if verts.shape[0] < 3:
        raise ValueError(f"A polygon requires at least 3 vertices. You provided {verts.shape[0]}.")

    # Apply the scale factor
    verts = verts * scale_factor

    # Poly3DCollection requires a list of polygons.
    # We wrap our N-vertex array in a single outer list to draw one face.
    polygon = Poly3DCollection([verts], alpha=alpha, facecolors=face_color, edgecolors=edge_color)

    # Add the polygon to the axis
    ax.add_collection3d(polygon)

if __name__ == "__main__":

    #Initialize plot

    ###3 coords systems used in this model, fuselage (a.c), wing and hinge

    theta_input = -45 #deg
    phi_input = 36 #deg

    wing_planform = Wing(
                np.array([[2, 0, 0], [0, 0, 10], [-1, 0, 10], [-1, 0, 0]]),
                theta_input,
                phi_input)

    wing_planform.add_point_load(PointLoad((200, 0, 0), (1, 0, 3.001)))
    wing_planform.add_point_load(PointLoad((200, 0, 0), (0.5, 0, 6.001)))

    def wing_loading(x):
        return -(x/5)**2 + 4
        #return x
    wing_planform.add_distributed_load(DistributedLoad((0, -1, 0), (0.5, 0, 0), (-1, 0, 10), wing_loading))

    wing_planform.discretize(0.01)

    wing_planform.angle = 60
    wing_planform.plot_wing()




    plt.show()

    """
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
    """
