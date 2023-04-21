'''Copyright © 2023 Landon Work. All rights reserved.'''
from typing import Callable, Optional, Union
import numpy as np

__all__ = [
    'Point',
    'lerp',
    'bezier_curve',
    'cubic_bezier_curve',
    'C1Spline',
    'SplineDeriv',
    'C1CubicSpline',
]

Point = np.ndarray

def is_point(p: Point) -> bool:
    return isinstance(p, Point) and p.shape == (2,)

# TODO: Make these matrix operations instead of normal arithmetic
def lerp(p1, p2, u):
    assert is_point(p1) and is_point(p2), f"{p1}, {p2}"
    assert 0 <= u and u <= 1, "invalid `u`"
    return p1 * (1-u) + p2 * u

def bezier_curve(p1, p2, p3, u):
    return lerp(lerp(p1, p2, u), lerp(p2, p3, u), u)

def cubic_bezier_curve(p1, p2, p3, p4, u = None):
    return lerp(bezier_curve(p1, p2, p3, u), bezier_curve(p2, p3, p4, u), u)

class C1Spline:
    def __init__(self, points):
        self.points = points
        self.n_curves = int(points.shape[0] - 1) // 2
        
    def add_curve(self, new_points):
        self.points = np.vstack([self.points, new_points])
        self.n_curves = int(self.points.shape[0] - 1) // 2

    def point_at_u(self, u):
        if u == self.n_curves:
            return self.points[-1,:]
        elif u > self.n_curves:
            raise ValueError("u: {u}")
        else:
            index = int(u) * 2
            val = float(u) % 1.
            return bezier_curve(*[self.points[ind, :] for ind in range(index, index+3)], val)

    def interpolate(self, step = 1e-1):
        u = np.arange(0., self.n_curves + 1e-8, step)
        return np.array([self.point_at_u(val) for val in u])

    def __call__(self, u):
        self.point_at_u(u)

    @staticmethod
    def from_vectors(velocities):
        """The name is a little vague, but this is how you can construct a spline
        for a CDF when all the vectors (velocities) are constrained to positive values

        All vector elements must be positive"""
        points = [np.zeros(2)]

        for i in range(velocities.shape[0] - 1):
            v0 = velocities[i, :]
            v1 = velocities[i+1, :]

            points.append(points[-1] + v0)
            points.append(points[-1] + v1)

        points = np.array(points)
        points = points / points[-1, :]

        return C1Spline(points)

class SplineDeriv:
    def __init__(self, spline):
        self.n_curves = spline.n_curves
        self.x_bounds = [spline.points[i, 0] for i in range(0, spline.n_curves * 2 + 1, 2)]
        self.points = spline.points
        self.curves = []
        self.M = np.array([
            [1, -2,  1],
            [0,  2, -2],
            [0,  0,  1]
        ])

        for i in range(spline.n_curves):
            ind = i * 2
            X = spline.points[ind:ind+3, 0].reshape(-1)
            Y = spline.points[ind:ind+3, 1].reshape(-1)

            def closure(u):
                dU = np.array([0, 1, 2*u, 3*u*u]).reshape((-1, 1))
                return np.matmul(np.matmul(Y, self.M), dU) / np.matmul(np.matmul(X, self.M), dU)

            self.curves.append(closure)

    def deriv_at_x(self, x):
        index = ((np.array(self.x_bounds) <= x).sum() - 1) * 2
        if index >= self.points.shape[0]:
            raise ValueError(f"x: {x}")
        # print('x:', x)
        # print('index:', index)
        p1 = self.points[index, :]
        p2 = self.points[index+1, :]
        p3 = self.points[index+2, :]
        # print('v1, v2:', p2[0] - p1[0], p3[0] - p2[0])

        # Set up quadratic equation: x = a * u^2 + b * u + c
        a =      p1[0] - 2 * p2[0] + p3[0]
        b = -2 * p1[0] + 2 * p2[0]
        c =      p1[0] - x
        # And solve for u
        u = quadratic_equation_pos(a, b, c)
        # u = quadratic_equation_neg(a, b, c)

        points = np.array([p1, p2, p3])
        X = points[:, 0].reshape((1, 3))
        Y = points[:, 1].reshape((1, 3))
        dU = np.array([0, 1, 2*u]).reshape((3, 1))
        dy_dx = np.matmul(np.matmul(Y, self.M), dU) / np.matmul(np.matmul(X, self.M), dU)

        return dy_dx

    def interpolate(self, step = 1e-1):
        x = np.arange(self.x_bounds[0], self.x_bounds[-1], step)
        interp = []
        for val in x:
            interp.append(self.deriv_at_x(val))

        return np.hstack([
            x.reshape((-1, 1)),
            np.array(interp).reshape((-1, 1))
        ])

    def __call__(self, x):
        return self.point_at_x(x)

class C1CubicSpline:
    def __init__(self, points):
        self.points = points
        self.n_curves = int(points.shape[0] - 1) // 3

    def add_curve(self, new_points):
        self.points = np.vstack([self.points, new_points])
        self.n_curves = int(self.points.shape[0] - 1) // 3

    def point_at_u(self, u):
        if u == self.n_curves:
            return self.points[-1,:]
        elif u > self.n_curves:
            raise ValueError(f"u: {u}")
        else:
            index = int(u) * 3
            val = float(u) % 1.
            return cubic_bezier_curve(*[self.points[ind, :] for ind in range(index, index+4)], val)

    def interpolate(self, step = 1e-1):
        u = np.arange(0, self.n_curves + 1e-8, step)
        return np.array([self.point_at_u(val) for val in u])

    def __call__(self, u):
        return self.point_at_u(u)

    @staticmethod
    def from_vectors(start, end, velocities, displacements):
        assert velocities.shape[0] == displacements.shape[0] + 1

        points = [start]
        for i in range(displacements.shape[0]):
            v0 = velocities[i, :]
            d = displacements[i, :]
            v1 = velocities[i+1, :]

            points.append(points[-1] + v0)
            points.append(points[-1] + d)
            points.append(points[-1] + v1)

        points = np.array(points)
        points = (points - start) / points[-1, :] * end

        return C1CubicSpline(points)

def quadratic_equation_pos(a, b, c):
    # print('a, b, c:', a, b, c)
    # print('Discriminant:', b * b - 4 * a * c)
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

def quadratic_equation_neg(a, b, c):
    # print('a, b, c:', a, b, c)
    # print('Discriminant:', b * b - 4 * a * c)
    return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
