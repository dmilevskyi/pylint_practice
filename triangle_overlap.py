"""Source: http://rosettacode.org/wiki/Determine_if_two_triangles_overlap#Python"""
from __future__ import print_function
import numpy as np


def check_tri_winding(tri, allow_reversed):
    """Check triangle winding"""
    trisq = np.ones((3, 3))
    trisq[:, 0:2] = np.array(tri)
    det_tri = np.linalg.det(trisq)
    if det_tri < 0.0:
        if allow_reversed:
            a = trisq[2, :].copy()  # pylint: disable=invalid-name
            trisq[2, :] = trisq[1, :]
            trisq[1, :] = a
        else:
            raise ValueError("triangle has wrong winding direction")
    return trisq


def tri_tri_2d(f_triangle, s_triangle, eps=0.0, allow_reversed=False, on_boundary=True):
    """Check is triangles collide"""
    # Trangles must be expressed anti-clockwise
    t1s = check_tri_winding(f_triangle, allow_reversed)
    t2s = check_tri_winding(s_triangle, allow_reversed)

    if on_boundary:
        # Points on the boundary are considered as colliding
        def chk_edge(parameter_array):
            return np.linalg.det(parameter_array) < eps
    else:
        # Points on the boundary are not considered as colliding
        def chk_edge(parameter_array):
            return np.linalg.det(parameter_array) <= eps

    # chk_edge = is_colliding(on_boundary, x, eps)

    # For edge E of trangle 1,
    for i in range(3):
        edge = np.roll(t1s, i, axis=0)[:2, :]

        # Check all points of trangle 2 lay on the external side of the edge E. If
        # they do, the triangles do not collide.
        if (
            chk_edge(np.vstack((edge, t2s[0])))
            and chk_edge(np.vstack((edge, t2s[1])))
            and chk_edge(np.vstack((edge, t2s[2])))
        ):
            return False

    # For edge E of trangle 2,
    for i in range(3):
        edge = np.roll(t2s, i, axis=0)[:2, :]

        # Check all points of trangle 1 lay on the external side of the edge E. If
        # they do, the triangles do not collide.
        if (
            chk_edge(np.vstack((edge, t1s[0])))
            and chk_edge(np.vstack((edge, t1s[1])))
            and chk_edge(np.vstack((edge, t1s[2])))
        ):
            return False

    # The triangles collide
    return True


if __name__ == "__main__":
    first_triangle = [[0, 0], [5, 0], [0, 5]]
    second_triangle = [[0, 0], [5, 0], [0, 6]]
    print(tri_tri_2d(first_triangle, second_triangle), True)

    first_triangle = [[0, 0], [0, 5], [5, 0]]
    second_triangle = [[0, 0], [0, 6], [5, 0]]
    print(tri_tri_2d(first_triangle, second_triangle, allow_reversed=True), True)

    first_triangle = [[0, 0], [5, 0], [0, 5]]
    second_triangle = [[-10, 0], [-5, 0], [-1, 6]]
    print(tri_tri_2d(first_triangle, second_triangle), False)

    first_triangle = [[0, 0], [5, 0], [2.5, 5]]
    second_triangle = [[0, 4], [2.5, -1], [5, 4]]
    print(tri_tri_2d(first_triangle, second_triangle), True)

    first_triangle = [[0, 0], [1, 1], [0, 2]]
    second_triangle = [[2, 1], [3, 0], [3, 2]]
    print(tri_tri_2d(first_triangle, second_triangle), False)

    first_triangle = [[0, 0], [1, 1], [0, 2]]
    second_triangle = [[2, 1], [3, -2], [3, 4]]
    print(tri_tri_2d(first_triangle, second_triangle), False)

    # Barely touching
    first_triangle = [[0, 0], [1, 0], [0, 1]]
    second_triangle = [[1, 0], [2, 0], [1, 1]]
    print(tri_tri_2d(first_triangle, second_triangle, on_boundary=True), True)

    # Barely touching
    first_triangle = [[0, 0], [1, 0], [0, 1]]
    second_triangle = [[1, 0], [2, 0], [1, 1]]
    print(tri_tri_2d(first_triangle, second_triangle, on_boundary=False), False)
