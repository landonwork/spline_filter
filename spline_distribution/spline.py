import math

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
        self.end_points = self.get_y_batched(np.unique(self.knots))

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

    # probably everything involved in sampling should get moved to the distribution file and
    # put into a CDF wrapper class or something
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
        

    def get_curve_index(self, y):
        return np.where((self.end_points[:-1] <= y) & (self.end_points[1:] > y))[0].tolist()

    def get_curve(self, ind):
        assert 0 <= ind < len(self.end_points), 'invalid index'
        assert self.degree == 3, 'only cubic curves implemented'
        knots_ind = ind + self.degree
        
        
        t1 = self.knots[knots_ind - 2]
        t2 = self.knots[knots_ind - 1]
        t3 = self.knots[knots_ind]
        t4 = self.knots[knots_ind + 1]
        t5 = self.knots[knots_ind + 2]
        t6 = self.knots[knots_ind + 3]

        p0 = self.points[knots_ind - 3]
        p1 = self.points[knots_ind - 2]
        p2 = self.points[knots_ind - 1]
        p3 = self.points[knots_ind]

        A = (t4 - t1) * (t4 - t2) * (t4 - t3)
        B = (t5 - t2) * (t4 - t2) * (t4 - t3)
        C = (t5 - t2) * (t5 - t3) * (t4 - t3)
        D = (t6 - t3) * (t5 - t3) * (t4 - t3)
        
        curve = Curve(
            -p0 / A + p1 * (1/A + 1/B + 1/C) - p2 * (1/B + 1/C + 1/D) + p3 / D,
            p0 * 3 * t4 / A
                - p1 * ( (t1 + 2*t4) / A + (t2 + t4 + t5) / B + (t3 + 2*t5) / C )
                + p2 * ( (2*t2 + t4) / B + (t2 + t3 + t5) / C + (2*t3 + t6) / D )
                - p3 * 3 * t3 /  D,
            -p0 * 3 * t4 * t4 / A
                + p1 * ( (2*t1*t4 + t4*t4) / A + (t2*t4 + t2*t5 + t4*t5) / B + (2*t3*t5 + t5*t5) / C )
                - p2 * ( (2*t2*t4 + t2*t2) / B + (t2*t3 + t2*t5 + t3*t5) / C + (2*t3*t6 + t3*t3) / D )
                + p3 * 3 * t3 * t3 / D,
            p0 * t4 * t4 * t4 / A
                - p1 * ( t1 * t4 * t4 / A + t2 * t4 * t5 / B + t3 * t5 * t5 / C )
                + p2 * ( t2 * t2 * t4 / B + t2 * t3 * t5 / C + t3 * t3 * t6 / D )
                - p3 * t3 * t3 * t3 / D
        )
        return curve

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

    def plot(self, segmented=True, **kwargs):
        knots = np.unique(self.knots)
        if segmented:
            for i in range(knots.shape[0] - 1):
                ts = np.linspace(knots[i], knots[i+1], 21)
                ys = self.get_y_batched(ts)
                plt.plot(ts, ys)
        else:
            ts = [knots[0]]
            for i in range(knots.shape[0] - 1):
                ts.extend(np.linspace(knots[i], knots[i+1], 21)[1:])
            plt.plot(ts, self.get_y_batched(np.array(ts)), **kwargs)


class Curve:
    def __init__(self, *coefficients):
        assert len(coefficients) == 4, 'we only support cubic curves at the moment'
        # assert len(coefficients) >= 1
        self.coefficients = coefficients
        self.degree = len(coefficients) - 1

    def __call__(self, x):
        ans = 0.
        for i, coef in enumerate(self.coefficients):
            ans += (x ** (self.degree - i)) * coef
        return ans

    def get_p_q(self, y):
        assert self.degree == 3, 'only cubic curves are supported'

        a = self.coefficients[0]
        b = self.coefficients[1]
        c = self.coefficients[2]
        d = self.coefficients[3] - y

        p = (3 * a * c - b * b) / (3 * a * a)
        q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a)
        return p, q

    def solve(self, y):
        # cbrt(-q/2 + sqrt((q/2)**2 + (p/3)**3)) + cbrt(-q/2 - sqrt((q/2)**2 + (p/3)**3)) - b / (3 * a)
        a = self.coefficients[0]
        b = self.coefficients[1]
        c = self.coefficients[2]
        d = self.coefficients[3] - y

        p, q = self.get_p_q(y)
        discriminant = (p/3)**3 + (q/2)**2
        if discriminant > 0.0:
            # one unique solution
            ans = np.cbrt(-q/2 + np.sqrt(discriminant)) + np.cbrt(-q/2 - np.sqrt(discriminant)) - b / (3 * a)
            return np.array([ans])
        elif discriminant == 0.0:  # TODO
            # two unique solutions
            assert False, 'I will implement this later'
        else:
            # three unique solutions
            # get the real and imaginary parts of the first cube root (which will be a complex number)
            real1 = -q / 2
            imag1 = np.sqrt(np.abs(discriminant))
            
            r = np.sqrt(np.square(real1) + np.square(imag1))
            theta = np.arctan2(imag1, real1)

            r_root = np.cbrt(r)
            theta_root = theta / 3
            # once we find the cube roots of the first complex number, the next part is easy.
            # because we know the real parts are the same and the imaginary parts are additive inverses (add to zero),
            # we can just pay attention to the real part and multiply by two.
            theta_roots = np.array([theta_root, theta_root + 2 * math.pi / 3, theta_root + 4 * math.pi / 3])
            ans = 2 * r_root * np.cos(theta_roots) - b / (3 * a)
            return ans

    def __repr__(self):
        chars = 'abcdef'
        params = []
        for i, coef in enumerate(self.coefficients):
            params.append(f'{chars[i]}={coef:.3f}')
        s = 'Curve(' + ', '.join(params) + ')'
        return s

    def __str__(self):
        return repr(self)