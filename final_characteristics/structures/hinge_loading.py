import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import cumulative_trapezoid
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import project_vector

class Wing():
    def __init__(self, geometry, theta_deg, phi_deg, alpha=0.0):
        #plot, created lazily in plot_wing so simulation-only runs don't open a figure per case
        self.fig = None
        self.ax = None
        self.geometry = geometry
        self.point_loads = []
        self.nonangled_point_loads = []
        self.nonangled_distributed_loads = []
        self.moments = []
        self.distributed_loads = []
        self._angle = 0
        self.hinge_axes = transform_axes(theta_deg, phi_deg)
        self.hinge_vector = self.hinge_axes[0]
        self.disc_res = 0.01

        reaction_forces = []
        reaction_forces.append((np.array([1, 0, 0]), np.zeros(3)))
        reaction_forces.append((np.array([0, 1, 0]), np.zeros(3)))
        reaction_forces.append((np.array([0, 0, 1]), np.zeros(3)))

        self.reaction_forces = reaction_forces #[(force, direction), ...] Magnitude unknown!!!!



        self.wing_axes = np.eye(3)

        self.shear_diagrams = {} #(x, y) means forces pointing towards x analyzed in the y direction
        self.moment_diagrams = {}
    @property
    def angle(self):
        return self._angle
    @angle.setter
    def angle(self, value):

        for point_load in self.nonangled_point_loads:
            point_load.dir = rotate_vectors_around_axis(self.hinge_vector, self._angle, point_load.dir)

        for dist_load in self.nonangled_distributed_loads:
            dist_load.dir = rotate_vectors_around_axis(self.hinge_vector, self._angle, dist_load.dir)

        self._angle = value

        for point_load in self.nonangled_point_loads:
            point_load.dir = rotate_vectors_around_axis(self.hinge_vector, -self._angle, point_load.dir)
        for dist_load in self.nonangled_distributed_loads:
            dist_load.dir = rotate_vectors_around_axis(self.hinge_vector, -self._angle, dist_load.dir)

        #rebuild from the standard axes so repeated angle assignments don't accumulate rotations
        self.wing_axes = rotate_vectors_around_axis(self.hinge_vector, self._angle, np.eye(3))
    def add_point_load(self, load, nonangled = False):
        if isinstance(load, PointLoad):
            if nonangled:
                load.dir = rotate_vectors_around_axis(self.hinge_vector, -self._angle, load.dir)
                self.nonangled_point_loads.append(load)
            else:
                self.point_loads.append(load)
        else: raise TypeError
    def add_distributed_load(self, distributed_load, nonangled=False):
        if isinstance(distributed_load, DistributedLoad):
            if nonangled:
                self.nonangled_distributed_loads.append(distributed_load)
                distributed_load.dir = rotate_vectors_around_axis(self.hinge_vector, -self._angle, distributed_load.dir)
            else:
                self.distributed_loads.append(distributed_load)
        else: raise TypeError
    def plot_wing(self, axes=True, point_forces=True, distributed_loads=True):

        if self.ax is None:
            self.fig = plt.figure(figsize=(8, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')

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

        if maxload == 0:
            maxload = 1

        if point_forces:

            for point_load in self.point_loads:
                point_dir_rotated = rotate_vectors_around_axis(self.hinge_vector, self.angle, point_load.dir)
                point_loc_rotated = rotate_vectors_around_axis(self.hinge_vector, self.angle, point_load.loc)
                plot_single_vector(self.ax, point_dir_rotated*point_load.magn/maxload, point_loc_rotated, point_load.color)

            for point_load in self.nonangled_point_loads:
                point_dir_rotated = rotate_vectors_around_axis(self.hinge_vector, self.angle, point_load.dir)
                point_loc_rotated = rotate_vectors_around_axis(self.hinge_vector, self.angle, point_load.loc)
                plot_single_vector(self.ax, point_dir_rotated*point_load.magn/maxload, point_loc_rotated, point_load.color)

        if distributed_loads:
            number_of_points = 10 #points for plotting distributed loads

            for load in self.distributed_loads:
                geo = []
                geo.append(load.loc)
                for i in range(number_of_points + 1):
                    x = np.sqrt(np.dot(load.load_dir, load.load_dir))/number_of_points*i
                    y = load.func(x)
                    load_dir_normalized = load.load_dir/np.sqrt(np.dot(load.load_dir, load.load_dir))
                    geo.append(load.loc + load.dir*y/1 + load_dir_normalized*x)
                geo.append(load.loc + load.load_dir)
                plot_polygon_3d(self.ax, rotate_vectors_around_axis(self.hinge_vector, self.angle, geo), face_color=load.color)

            for load in self.nonangled_distributed_loads:
                geo = []
                geo.append(load.loc)
                for i in range(number_of_points + 1):
                    x = np.sqrt(np.dot(load.load_dir, load.load_dir)) / number_of_points * i
                    y = load.func(x)
                    load_dir_normalized = load.load_dir / np.sqrt(np.dot(load.load_dir, load.load_dir))
                    geo.append(load.loc + load.dir * y/1 + load_dir_normalized * x)
                geo.append(load.loc + load.load_dir)
                plot_polygon_3d(self.ax, rotate_vectors_around_axis(self.hinge_vector, self.angle, geo), face_color=load.color)

        plot_axes(self.ax)

        if axes:
            plot_axes(self.ax, axis=self.wing_axes)




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
    def discretize(self, distributed_mesh_size=100):
        ###Numpy array with discretized force positions
        res = self.disc_res
        discretized_forces = []
        discretized_positions = []

        for load in self.point_loads:
            discretized_positions.append(np.round(load.loc / res) * res)
            discretized_forces.append(load.dir*load.magn)

        for load in self.nonangled_point_loads:
            discretized_positions.append(np.round(load.loc / res) * res)
            discretized_forces.append(load.dir*load.magn)

        for load in self.distributed_loads + self.nonangled_distributed_loads:
            load_dir_magn = np.sqrt(np.dot(load.load_dir,load.load_dir))
            load_dir_norm = load.load_dir/load_dir_magn
            dx = load_dir_magn/(distributed_mesh_size - 1)
            for i in range(distributed_mesh_size):
                x = dx*i
                y = load.func(x)
                #trapezoid weights: samples are spaced dx apart, end points carry half a cell
                weight = dx/2 if i in (0, distributed_mesh_size - 1) else dx
                discretized_forces.append(load.dir * y * weight)
                discretized_positions.append(np.round((load.loc + load_dir_norm*x) / res) * res)

        self.discretized_forces = np.array(discretized_forces).T
        self.discretized_positions = np.array(discretized_positions).T
    def create_loading_diagram(self, force_direction="y", path_direction="x", plot=False):


        force_index = ["x", "y", "z"].index(force_direction)
        path_index = ["x", "y", "z"].index(path_direction)



        arr = np.vstack((self.discretized_forces[force_index], self.discretized_positions[path_index]))


        # Introduce reaction force

        #rforce = np.array([-np.sum(arr[0, :]), 0])
        #arr = np.column_stack((arr, rforce))


        #sort forces by position; coincident forces are summed by the accumulation loop below
        sort_indices = np.argsort(arr[1, :])
        arr = np.round(arr[:, sort_indices], decimals=5)

        x_ax = np.round(np.arange(min(arr[1][0], 0), max(arr[1, -1], 0) + self.disc_res, self.disc_res), decimals=5).tolist()
        y_ax = np.zeros_like(x_ax)

        for i in range(len(arr[0])):
            start_idx = x_ax.index(arr[1][i])
            y_ax[start_idx:] += arr[0, i]

        reaction_force = -np.sum(arr[0])

        y_ax[x_ax.index(0):] += reaction_force

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax1.plot(x_ax, y_ax, color='b')

        self.shear_diagrams[(force_direction, path_direction)] = (x_ax, y_ax)

        return reaction_force
    def create_moment_diagram(self, axis="y", plot=False):
        #get info from the shear diagrams and if it doesn't exist, create it
        #eg for y axis we need force in x dir towards z and vice versa
        lst = ["x", "y", "z"]
        lst.remove(axis)

        self.create_loading_diagram(force_direction=lst[0], path_direction=lst[1])
        shear_1 = self.shear_diagrams[(lst[0], lst[1])]

        self.create_loading_diagram(force_direction=lst[1], path_direction=lst[0])
        shear_2 = self.shear_diagrams[(lst[1], lst[0])]

        moment_1 = cumulative_trapezoid(shear_1[1], shear_1[0], initial=0)
        moment_2 = cumulative_trapezoid(shear_2[1], shear_2[0], initial=0)

        #One of the two moments must be flipped
        flipslst = {"x": "y", "y": "z", "z": "x"}
        if lst[0] == flipslst[axis]:
            moment_1 *= -1
        elif lst[1] == flipslst[axis]:
            moment_2 *= -1

        #The hinge support sits at coordinate 0 along each path, so its reaction moment
        #enters the diagram as a step there rather than a shift of the whole curve;
        #this keeps both free ends at zero moment even when loads straddle the support.
        r_moment_1 = -moment_1[-1]
        moment_1[shear_1[0].index(0):] += r_moment_1
        r_moment_2 = -moment_2[-1]
        moment_2[shear_2[0].index(0):] += r_moment_2

        if plot:
            #moment_1 and moment_2 are functions of different path coordinates,
            #so each is plotted against its own shear diagram's axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax1.plot(shear_1[0], shear_1[1], color='b', label=f"shear of force {lst[0]} in direction {lst[1]}", alpha=0.4)
            ax1.plot(shear_2[0], shear_2[1], color="r", label=f"shear of force {lst[1]} in direction {lst[0]}", alpha=0.4)
            m = max(np.abs(shear_1[1]).max(), np.abs(shear_2[1]).max())
            ax1.set_ylim(-1.25*m, 1.25*m)
            ax1.legend()
            ax1.axhline(0, color='black', linestyle=':', linewidth=1.5)

            ax2.plot(shear_1[0], moment_1, color='b', label=f"moment of force {lst[0]} in direction {lst[1]}", alpha=0.4)
            ax2.plot(shear_2[0], moment_2, color="r", label=f"moment of force {lst[1]} in direction {lst[0]}", alpha=0.4)
            m = max(np.abs(moment_1).max(), np.abs(moment_2).max())
            ax2.set_ylim(-1.25 * m, 1.25 * m)
            ax2.legend()
            ax2.axhline(0, color='black', linestyle=':', linewidth=1.5)

        #r_moment_1 + r_moment_2 equals the applied moment about the axis; negate so the
        #returned value is the reaction moment, matching create_loading_diagram's convention
        return -(r_moment_1 + r_moment_2)
    def resolve_reactions(self):

        ###RANDOM AHH REACTION FORCE
        #self.reaction_forces.append((self.hinge_vector, [0, 0, 10]))


        #equilibrium: sum of reactions = -(sum of applied forces)
        reaction_force_res = -np.sum(self.discretized_forces, axis=1).reshape(3, 1)
        reaction_force_mat = np.array([[0], [0], [0]])

        for tup in self.reaction_forces:
            tup = np.array(tup[0]).reshape(3, 1)
            reaction_force_mat = np.append(reaction_force_mat, tup, axis=1)
        reaction_force_mat = np.delete(reaction_force_mat, 0, axis=1)

        final_matrix = reaction_force_mat
        final_reactions = reaction_force_res

        ###THE PLAN!!!
        #Each reaction force (even the ones at the origin) has a position and a direction, MAGNITUDE MUST BE FOUND!!!
        #Each reaction force also produces a reaction moment, the equations of which will have to be built and resolved
        #EQUATIONS FOR REACTION FORCES are easy because you don't care about which point you're looking from.
        ###FOR NOW THIS IS ALL DONE
        """
        For moments, you need to take moments around different points
        Every position in self.discretized_positions must be expressed relative to the point you're taking moments around
        Then for that position generate 3 moment equations. FIGURE OUT HOW TO DO THAT
        """

        #This code now finds the right hand side for each moment equation. Now it's just time for the left hand side!

        already_checked = []

        for tup in self.reaction_forces:

            if not any(np.array_equal(tup[1], x) for x in already_checked):

                already_checked.append(tup[1])
                current_origin = np.array(tup[1])
                #find the vector positions relative to the new origin
                relative_positions = self.discretized_positions - current_origin.reshape(3, 1)
                #total applied moment about the new origin; reactions must balance it
                mx = self.discretized_forces[2, :]*relative_positions[1, :] - self.discretized_forces[1, :]*relative_positions[2, :]
                my = self.discretized_forces[0, :]*relative_positions[2, :] - self.discretized_forces[2, :]*relative_positions[0, :]
                mz = self.discretized_forces[1, :]*relative_positions[0, :] - self.discretized_forces[0, :]*relative_positions[1, :]
                total_moments = -np.array([np.sum(mx), np.sum(my), np.sum(mz)]).reshape(3, 1)

                curr_matrix = np.array([0, 0, 0]).reshape(3, 1)
                for rtup in self.reaction_forces:
                    curr_reaction_force = rtup[0]
                    curr_reaction_position_relative = rtup[1] - current_origin
                    rmx = curr_reaction_force[2] * curr_reaction_position_relative[1] - curr_reaction_force[1] * curr_reaction_position_relative[2]
                    rmy = curr_reaction_force[0] * curr_reaction_position_relative[2] - curr_reaction_force[2] * curr_reaction_position_relative[0]
                    rmz = curr_reaction_force[1] * curr_reaction_position_relative[0] - curr_reaction_force[0] * curr_reaction_position_relative[1]
                    curr_reaction_force_moments = np.array([rmx, rmy, rmz])
                    curr_matrix = np.append(curr_matrix, curr_reaction_force_moments.reshape(3, 1), axis=1)
                curr_matrix = np.delete(curr_matrix, 0, axis=1)
                if np.dot(current_origin, current_origin) == 0:
                    #moments about the support point itself can't be balanced by forces
                    #acting there, so they are excluded from the solve
                    origin_matrix = curr_matrix
                    origin_reactions = total_moments
                else:
                    final_matrix = np.vstack((final_matrix, curr_matrix))
                    final_reactions = np.vstack((final_reactions, total_moments))

        ###attempt to solve
        x, residuals, rank, s = np.linalg.lstsq(final_matrix, final_reactions, rcond=None)

        self.reaction_force_magnitudes = x

        """
        print(self.discretized_forces)
        print(self.reaction_forces)
        print(self.reaction_force_magnitudes)

        for i in range(len(self.reaction_forces)):
            self.discretized_positions = np.hstack((self.discretized_positions,  np.round(np.array(self.reaction_forces[i][1]).reshape(3,1) / self.disc_res) * self.disc_res))
            self.discretized_forces = np.hstack((self.discretized_forces,  np.array(self.reaction_forces[i][1]).reshape(3,1) * self.reaction_force_magnitudes[i]))

        print(self.discretized_positions)
        print(self.discretized_forces)
        """


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
    Builds the hinge coordinate system.

    Returns a 3x3 matrix whose ROWS are the transformed X, Y and Z unit axes,
    matching the row convention used for wing_axes elsewhere in this file.
    Row i is the standard axis i rotated first by -phi_deg around Z, then by
    -theta_deg around Y.

    For theta = -45 and phi = 35.26439 (= atan(1/sqrt(2))), row 0 -- used as
    the hinge vector -- is the body diagonal (1, -1, -1)/sqrt(3), about which
    a 120 deg rotation tilts the wing span through 90 deg.

    Parameters:
    theta_deg (float): Rotation angle around Y axis in degrees (negated, applied second).
    phi_deg (float): Rotation angle around Z axis in degrees (negated, applied first).

    Returns:
    transformed_axes (np.ndarray): A 3x3 matrix where each ROW represents
                                   the new X, Y, and Z unit vectors.
    """
    # Convert angles from degrees to radians
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    R_z = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [+np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])

    # R_z(phi) @ R_y(theta) as a matrix; its rows are the standard axes rotated
    # by R_y(-theta) @ R_z(-phi), which is the convention described in the docstring
    R_combined = np.dot(R_z, R_y)

    return R_combined
def transform_vector(vector, native_axis, axis_transform_to):

    if native_axis is axis_transform_to:
        return vector

    #Both the native axis and the axis to transform to must be relative to the origin
    vector_transform_1 = native_axis[0]*vector[0] + native_axis[1]*vector[1] + native_axis[2]*vector[2]


    x_comp = project_vector(vector_transform_1, axis_transform_to[0])
    x_magn = np.dot(x_comp, axis_transform_to[0])/(np.sqrt(np.dot(axis_transform_to[0], axis_transform_to[0])))


    y_comp = project_vector(vector_transform_1, axis_transform_to[1])
    y_magn = np.dot(y_comp, axis_transform_to[1])/(np.sqrt(np.dot(axis_transform_to[1], axis_transform_to[1])))

    z_comp = project_vector(vector_transform_1, axis_transform_to[2])
    z_magn = np.dot(z_comp, axis_transform_to[2])/(np.sqrt(np.dot(axis_transform_to[2], axis_transform_to[2])))

    return np.array([x_magn, y_magn, z_magn])
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
def plot_single_vector(ax, vector, origin=(0, 0, 0), color='b', label=None, alpha=1):
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
              color=color, arrow_length_ratio=0.5, linewidth=2, label=label, alpha=alpha)
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
    plot_single_vector(ax, axis[0], color='r', label='Vector 1', alpha=0.1)
    plot_single_vector(ax, axis[1], color='g', label='Vector 1', alpha=0.1)
    plot_single_vector(ax, axis[2], color='b', label='Vector 1', alpha=0.1)
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


def run_wing_sim(angle, v):

    wing_planform = Wing(
        np.array([[0.25 * root_chord, 0, 0], [0.25 * root_chord * taper, 0, halfspan_hinge],
                  [-0.75 * root_chord * taper, 0, halfspan_hinge], [-0.75 * root_chord, 0, 0]]),
        theta_input,
        phi_input)

    v_actual = v

    def weight(x):
        return 145

    def wing_loading(x):
        return (-(x/7)**2 + 3)*180/v_cruise**2*v_actual**2

    #lift acts on the quarter-chord line, which for this geometry lies at x = 0 along the whole span
    wing_planform.add_distributed_load(
        DistributedLoad((0, -1, 0), (0, 0, 0), (0, 0, halfspan_hinge), wing_loading))

    wing_planform.add_distributed_load(
        DistributedLoad((0, 1, 0), (-1, 0, 0), (-0, 0, halfspan_hinge), weight, color="blue"), nonangled=True)

    ###PROPELLER THRUST FORCES (wing-fixed) AND WEIGHTS (gravity-fixed, hence nonangled)
    wing_planform.add_point_load(PointLoad((propeller_thrust, 0, 0), thruster_position_1, color="orange"))
    wing_planform.add_point_load(PointLoad((propeller_thrust, 0, 0), thruster_position_2, color="orange"))
    wing_planform.add_point_load(PointLoad((0, propeller_weight, 0), thruster_position_1, color="brown"), nonangled=True)
    wing_planform.add_point_load(PointLoad((0, propeller_weight, 0), thruster_position_2, color="brown"), nonangled=True)

    wing_planform.angle = angle

    wing_planform.discretize()

    rx = wing_planform.create_loading_diagram(force_direction="x", path_direction="z")
    ry = wing_planform.create_loading_diagram(force_direction="y", path_direction="z")
    rz = wing_planform.create_loading_diagram(force_direction="z", path_direction="y")
    mx = wing_planform.create_moment_diagram("x")
    my = wing_planform.create_moment_diagram("y")
    mz = wing_planform.create_moment_diagram("z")

    resultant_force = np.array([rx, ry, rz])

    resultant_force_n = resultant_force/np.sqrt(np.dot(resultant_force, resultant_force))

    #plot_single_vector(wing_planform.ax, resultant_force_n, color="purple", alpha=1)

    #both reactions are computed in the wing frame; express them in the global frame
    resultant_force = transform_vector(resultant_force, wing_planform.wing_axes, np.eye(3))
    resultant_moment = transform_vector(np.array([mx, my, mz]), wing_planform.wing_axes, np.eye(3))

    #plot_axes(wing_planform.ax, wing_planform.hinge_axes)

    return resultant_force, resultant_moment



if __name__ == "__main__":

    #Initialize plot

    ###3 coords systems used in this model, fuselage (a.c), wing and hinge

    theta_input = -45 #deg
    phi_input = 35.26438968 #deg

    root_chord = 3 #m
    taper = 0.8
    halfspan_hinge = 10 #From the point where the hinge starts
    #assuming hinge is positioned at the quarter chord

    thruster_position_1 = (0.5, 0, 3)
    thruster_position_2 = (0.5, 0, 6)

    v_cruise = 81
    v_actual = 0 #The actual velocity you are expecting during whatever config you're testing out
    propeller_thrust = 240 #N
    propeller_weight = (7.31 + 48.09) * 9.81

    angle_cases = [0,60,120]
    v_cases = [(120 - angle)**2/251.221 for angle in angle_cases]

    forces_lst = []
    moments_lst = []

    for i in range(len(angle_cases)):
        force, moment = run_wing_sim(angle_cases[i], v_cases[i])
        forces_lst.append(force)
        moments_lst.append(moment)

    np.savetxt("r_moments.txt", np.array(moments_lst))
    np.savetxt("r_forces.txt", np.array(forces_lst))