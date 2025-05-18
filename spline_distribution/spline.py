import matplotlib.pyplot as plt
import numpy as np


def basis_array(knots, degree, t):
    """Evaluates the B-spline basis function N_{i,k} at t (vectorized version)

    Parameters:
    ===========
    knots: 1-D numpy array of knots
    degree: degree of the spline
    t: Value of t (scalar for now)
    """
    n = knots.shape[0] - 1 - degree

    # Find the knot index `i`
    i = np.searchsorted(knots, t) - 1

    basis = np.zeros(n + 1)  # 1 extra zero for the far-right values
    if i >= n or i < degree:
        return basis[:-1]

    basis[i] = 1

    num1 = t - knots[i-degree:i+1]
    num2 = knots[i+1:i+degree+2] - t
    for r in range(1, degree + 1):
        denom = knots[i:i+r+2] - knots[i-r:i+2]
        denom = np.where(denom == 0, np.inf, denom)
        basis[i-r:i+1] = num1[-r-1:] / denom[:-1] * basis[i-r:i+1] + num2[:r+1] / denom[1:] * basis[i-r+1:i+2]

    return basis[:-1]


class BSpline1D:
    def __init__(self, points, knots, degree):
        assert len(points.shape) == 1
        assert len(knots.shape) == 1
        assert points.shape[0] > degree
        assert knots.shape[0] == points.shape[0] + degree + 1

        self.n = points.shape[0]
        self.points = points
        self.knots = knots
        self.degree = degree

    def get_interior_points(self):
        return self.points[self.degree-1:1-self.degree]

    @staticmethod
    def clamped(interior_points, interior_knots, degree):
        assert interior_points.shape[0] > 1
        assert degree > 0
        assert interior_knots.shape[0] == interior_points.shape[0] + degree - 1

        begin_point = interior_points[0]
        end_point = interior_points[-1]

        begin_knot = interior_knots[0]
        end_knot = interior_knots[-1]

        # m == n + degree - 1
        assert interior_knots.shape[0] == interior_points.shape[0] + degree - 1
        points = np.concatenate([[begin_point for _ in range(degree - 1)], interior_points, [end_point for _ in range(degree - 1)]])
        # N == n + 2 * (degree - 1)
        # m == N - (degree - 1)
        assert interior_knots.shape[0] == points.shape[0] - (degree - 1)
        knots = np.concatenate([[begin_knot for _ in range(degree)], interior_knots, [end_knot for _ in range(degree)]])
        # M == N + degree + 1
        assert knots.shape[0] == points.shape[0] + degree + 1

        return BSpline1D(points, knots, degree)

    @staticmethod
    def clamped_uniform(interior_points, degree, min_t=0, max_t=1):
        assert interior_points.shape[0] > 1
        assert degree > 0

        interior_knots = np.linspace(min_t, max_t, interior_points.shape[0] + degree - 1)

        return BSpline1D.clamped(interior_points, interior_knots, degree)

    def get_deriv(self):
        return BSpline1D(np.diff(self.points), self.knots[1:-1], self.degree-1)

    def get_y(self, t):
        basis = self.get_basis(t)
        return (basis * self.points).sum()

    def get_y_batched(self, t):
        basis = self.get_basis(t)
        return (basis * self.points.reshape((1, -1))).sum(axis=1)

    def get_basis(self, t):
        if isinstance(t, (int, float)):
            return basis_array(self.knots, self.degree, t)
        elif isinstance(t, np.ndarray):
            return self.basis_batched(t)
        else:
            raise TypeError

    def basis_batched(self, t):
        """Evaluates the B-spline basis function N_{i,k} at t (vectorized version)

        Parameters:
        ===========
        knots: 1-D numpy array of knots
        degree: degree of the spline
        t: Values of t (1-D numpy array)
        """
        # Find the knot index `i`
        i = np.searchsorted(self.knots, t) - 1
        rows = np.where((i < self.n - self.degree) | (i >= self.degree))

        # 1 row per value
        basis = np.zeros((t.shape[0], self.knots.shape[0]))
        for row in rows:
            basis[row, i[row]] = 1

        num1 = t.reshape((-1, 1)) - self.knots.reshape((1, -1))
        num2 = self.knots.reshape((1, -1)) - t.reshape((-1, 1))
        for r in range(1, self.degree + 1):
            denom = (self.knots[r:] - self.knots[:-r]).reshape((1, -1))
            denom = np.where(denom == 0, np.inf, denom).reshape((1, -1))
            basis[:, :-r-1] = (
                num1[:, :-r-1] / denom[:, :-1] * basis[:, :-r-1]
                + num2[:, r+1:] / denom[:, 1:] * basis[:, 1:-r]
            )

        return basis[:, :self.n]

    def plot(self):
        knots = np.unique(self.knots)
        for i in range(knots.shape[0] - 1):
            ts = np.linspace(knots[i], knots[i+1], 21)
            ys = self.get_y_batched(ts)
            plt.plot(ts, ys)
            plt.show()
