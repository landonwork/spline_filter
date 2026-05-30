import numpy as np
import matplotlib.pyplot as plt

from .spline import BSpline1D


class BSplineCDF(BSpline1D):
    def __init__(self, points, knots, degree):
        super().__init__(points, knots, degree)

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
        points = np.concatenate([
            [begin_point for _ in range(degree - 1)],
            interior_points,
            [end_point for _ in range(degree - 1)]
        ])
        # N == n + 2 * (degree - 1)
        # m == N - (degree - 1)
        assert interior_knots.shape[0] == points.shape[0] - (degree - 1)
        knots = np.concatenate([[begin_knot for _ in range(degree)], interior_knots, [end_knot for _ in range(degree)]])
        # M == N + degree + 1
        assert knots.shape[0] == points.shape[0] + degree + 1

        return BSplineCDF(points, knots, degree)

    @staticmethod
    def clamped_uniform(interior_points, degree, min_t=0, max_t=1):
        assert interior_points.shape[0] > 1
        assert degree > 0

        interior_knots = np.linspace(min_t, max_t, interior_points.shape[0] + degree - 1)

        return BSplineCDF.clamped(interior_points, interior_knots, degree)

    def sample(self, size=1):
        assert self.degree == 3, 'only for CDFs'

        ys = np.random.rand(size)
        ts = []
        for y in ys:
            curve_ind = self.get_curve_index(y)[0]
            curve = self.get_curve(curve_ind)
            solutions = curve.solve(y)
            if solutions.shape[0] == 1:
                t = solutions[0]
            else:
                t = solutions[np.where((self.knots[curve_ind + self.degree] <= solutions) & (solutions < self.knots[curve_ind + self.degree + 1]))][0]
            ts.append(t)

        if size == 1:
            return ts[0]
        else:
            return np.array(ts)

    def plot_sf(self, segmented=True, ax=None, **kwargs):
        knots = np.unique(self.knots)
        
        if segmented:
            for i in range(knots.shape[0] - 1):
                ts = np.linspace(knots[i], knots[i+1], 21)
                ys = 1 - self.get_y_batched(ts)
                if ax is None:
                    plt.plot(ts, ys)
                else:
                    ax.plot(ts, ys)
        else:
            ts = [knots[0]]
            for i in range(knots.shape[0] - 1):
                ts.extend(np.linspace(knots[i], knots[i+1], 21)[1:])
            ys = 1 - self.get_y_batched(np.array(ts))
            if ax is None:
                return plt.plot(ts, ys, **kwargs)
            else:
                return ax.plot(ts, ys, **kwargs)
