
import unittest
import numpy as np
from hinge_loading import (
    Wing,
    PointLoad,
    DistributedLoad,
    transform_axes,
    rotate_vectors_around_axis
)


class TestMathTransforms(unittest.TestCase):

    def test_rotate_vector_around_axis(self):
        testvec = np.array([1, 0, 0])
        testvec2 = np.array([-1, 0, 0])
        axis1 = np.array([0, 0, 1])
        axis2 = np.array([0, 0, -1])
        axis3 = np.array([1, 0, 0])
        axis4 = np.array([0, 0, 10000])

        np.testing.assert_array_almost_equal(rotate_vectors_around_axis(axis1, 90, testvec), np.array([0, 1, 0]), decimal=5,)
        np.testing.assert_array_almost_equal(rotate_vectors_around_axis(axis2, 90, testvec), np.array([0, -1, 0]), decimal=5,)
        np.testing.assert_array_almost_equal(rotate_vectors_around_axis(axis2, 90, testvec2), np.array([0, 1, 0]), decimal=5,)
        np.testing.assert_array_almost_equal(rotate_vectors_around_axis(axis1, 90, testvec2), np.array([0, -1, 0]), decimal=5,)
        np.testing.assert_array_almost_equal(rotate_vectors_around_axis(axis3, 90, testvec), np.array([1, 0, 0]), decimal=5,)
        np.testing.assert_array_almost_equal(rotate_vectors_around_axis(axis4, 90, testvec), np.array([0, 1, 0]), decimal=5, )
        np.testing.assert_array_almost_equal(rotate_vectors_around_axis(axis4, 0, testvec), np.array([1, 0, 0]), decimal=5, )

    def test_transform_axes_identity(self):
        # With 0 rotation, the axes should remain the standard identity matrix
        axes = transform_axes(0, 0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(axes, expected, decimal=5)


class TestLoadClasses(unittest.TestCase):
    def test_point_load_magnitude_and_direction(self):
        # A 3-4-5 triangle force vector
        force = np.array([300, 400, 0])
        loc = np.array([1, 1, 1])
        load = PointLoad(force, loc)

        self.assertEqual(load.magn, 500.0, "Magnitude calculation is incorrect.")

        expected_dir = np.array([0.6, 0.8, 0.0])
        np.testing.assert_array_almost_equal(load.dir, expected_dir, decimal=5,
                                             err_msg="Direction vector is not properly normalized.")


class TestWingStatics(unittest.TestCase):
    def setUp(self):
        # Initialize a simple flat wing aligned with standard axes
        geom = np.array([[2, 0, 0], [0, 0, 10], [-1, 0, 10], [-1, 0, 0]])
        self.wing = Wing(geom, theta_deg=0, phi_deg=0)
        # Suppress plotting during testing to keep tests fast and headless
        import matplotlib.pyplot as plt
        plt.ioff()

    def test_add_loads(self):
        load1 = PointLoad([100, 0, 0], [0, 0, 5])
        load2 = PointLoad([0, 100, 0], [0, 0, 5])
        self.wing.add_point_load(load1)
        self.wing.add_point_load(load2, nonangled=True)

        self.assertEqual(len(self.wing.point_loads), 1)
        self.assertEqual(len(self.wing.nonangled_point_loads), 1)

    def test_static_equilibrium_shear(self):
        # Apply a single 100N force in the +X direction
        force_val = 100
        self.wing.add_point_load(PointLoad([force_val, 0, 0], [0, 0, 5]))
        self.wing.discretize()

        # The reaction force at the root should be exactly -100N to maintain equilibrium
        reaction_force = self.wing.create_loading_diagram(force_direction="x", path_direction="z")

        self.assertAlmostEqual(reaction_force, -force_val, places=4,
                               msg="Reaction force does not balance the applied loads.")

    def test_static_equilibrium_multiple_loads(self):
        # Apply conflicting forces
        self.wing.add_point_load(PointLoad([150, 0, 0], [0, 0, 5]))
        self.wing.add_point_load(PointLoad([-50, 0, 0], [0, 0, 8]))
        self.wing.discretize()

        # Net applied force is +100 in X. Reaction should be -100.
        reaction_force = self.wing.create_loading_diagram(force_direction="x", path_direction="z")
        self.assertAlmostEqual(reaction_force, -100.0, places=4)


if __name__ == '__main__':
    unittest.main()
