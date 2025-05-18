"""
Simple Bayesian probability distributions, one made from a quadratic
bezier spline and one made from a cubic B-spline filter (cross your
fingers). This experiment is to see if splines that learn have the same
drawbacks as other Bayesian distributions which almost always get narrower as
they learn. It would be nice if we could see these distributions "unlearn".
"""

import numpy as np
import tensorflow as tf

@tf.function
def tf_basis_batched(knots, degree, t):
    knots = tf.reshape(knots, [1, -1])
    t = tf.reshape(t, [-1, 1])
    basis = tf.where(
        (knots[:, :-1] <= t) & (t < knots[:, 1:]),
        1.0,
        0.0
    )

    num1 = t - knots
    num2 = knots - t
    for r in range(1, degree + 1):
        denom = knots[:, r:] - knots[:, :-r]
        denom = tf.where(denom == 0, np.inf, denom)
        basis = (
            num1[:, :-r-1] / denom[:, :-1] * basis[:, :-1]
            + num2[:, r+1:] / denom[:, 1:] * basis[:, 1:]
        )

    return basis

class CubicBSplineDistribution:
def __init__(self, n, dtype=tf.float32):
    """
    Parameters:
    ===========
    :param n: number of interior knots
    """
    assert isinstance(n, int) and n > 1
    if n == 1:
        print('Warning: density with n == 1 is static')

    self.n = n
    self.m = n + 2  # m = n + degree - 1 = n + 3 - 1
    self.N = n + 4  # N = n + 2 * (degree - 1) = n + 2 * (3 - 1)
    self.M = n + 8  # M == N + degree + 1 = n + 2 * (degree - 1) + degree + 1 = n + 3 * degree - 1 = n + 3 * 3 - 1
    self.degree = 3

    interior_knots = np.linspace(0.0, 1.0, self.m)
    self.interior_knots = tf.convert_to_tensor(interior_knots, dtype=dtype)
    self.knots = tf.convert_to_tensor(np.concatenate([[0.0] * 3, interior_knots, [1.0] * 3]), dtype=dtype)
    self.deriv_knots = tf.convert_to_tensor(np.concatenate([[0.0] * 2, interior_knots, [1.0] * 2]), dtype=dtype)

    def neg_log_density(self):
        @tf.function
        def neg_log_density_inner(t_true, deriv_interior_points):  # t_true should be 1-D, deriv_interior_points should be 2-D
            n_rows = tf.shape(deriv_interior_points)[0]
            zeros = tf.reshape(tf.zeros(n_rows), [n_rows, 1])
            deriv_points = tf.concat(
                [
                    zeros,
                    zeros,
                    deriv_interior_points,
                    zeros,
                    zeros
                ],
                axis=1)
            basis = tf_basis_batched(self.deriv_knots, self.degree - 1, t_true)
            density = tf.reduce_sum(deriv_points * basis, axis=1)
            return -tf.math.log(density)

        return neg_log_density_inner

    def neg_log_cdf(self):
        @tf.function
        def neg_log_cdf_inner(t_pred, over_pred_vec, deriv_interior_points):
            n_rows = tf.shape(deriv_interior_points)[0]
            zeros = tf.reshape(tf.zeros(n_rows), [n_rows, 1])
            ones = tf.reshape(tf.ones(n_rows), [n_rows, 1])
            points = tf.concat(
                [
                    zeros,
                    zeros,
                    zeros,
                    tf.cumsum(deriv_interior_points, axis=-1),
                    ones,
                    ones,
                ],
                axis=1)
            basis = tf_basis_batched(self.knots, self.degree, t_pred)
            cdf = tf.reduce_sum(points * basis, axis=1)
            loss = -tf.math.log(tf.where(
                over_pred_vec > 0.5, # If we overpredicted t,
                cdf, # then loss is neg log of the chance of t_true < t_pred
                1 - cdf # If not, then we underpredicted and loss is neg log of the chance of t_true > t_pred
            ))
            return loss

        return neg_log_cdf_inner

# from tqdm import tqdm, trange

class SimpleDensityEstimator:
    def __init__(self, n=5):
        assert n > 1
        print('Warning: density with n == 1 is static')
        self.n = n
        self.dist = CubicBSplineDistribution(n)
        self.pdf_loss_fn = self.dist.neg_log_density()
        self.cdf_loss_fn = self.dist.neg_log_cdf()
        self.logits = tf.Variable(tf.random.uniform([n - 1]))
        self.optimizer = tf.keras.optimizers.Adam()

    def train(self, t_true, n_epochs=10, batch_size=128):
        for epoch in range(n_epochs):
            print(f'Epoch {epoch}')
            for i in range(t_true.shape[0] // batch_size):
                with tf.GradientTape() as tape:
                    deriv_interior_points = tf.reshape(tf.nn.softmax(self.logits), [1, -1])
                    loss = self.pdf_loss_fn(t_true[i*batch_size:(i+1)*batch_size], deriv_interior_points)
                grads = tape.gradient(loss, [self.logits])
                self.optimizer.apply_gradients(zip(grads, [self.logits]))

    def train_cdf(self, t_pred, over_pred_vec, n_epochs=10, batch_size=128):
        for epoch in range(n_epochs):
            print(f'Epoch {epoch}')
            for i in range(t_pred.shape[0] // batch_size):
                with tf.GradientTape() as tape:
                    deriv_interior_points = tf.reshape(tf.nn.softmax(self.logits), [1, -1])
                    loss = self.cdf_loss_fn(
                        t_pred[i*batch_size:(i+1)*batch_size],
                        over_pred_vec[i*batch_size:(i+1)*batch_size],
                        deriv_interior_points)
            grads = tape.gradient(loss, [self.logits])
            self.optimizer.apply_gradients(zip(grads, [self.logits]))

    def area(self, t_lo=0.0, t_hi=1.0):
        assert t_lo >= 0.0 and t_hi <= 1.0
        deriv_knots = self.dist.deriv_knots.numpy()
        included = deriv_knots[np.where((deriv_knots > t_lo) & (deriv_knots < t_hi))].tolist()
        included = [t_lo, *included, t_hi]

        i = np.searchsorted(deriv_knots, t_lo, side='right') - 1

        value = 0.0
        for j in range(len(included) - 1):
            value += self._area_integral(deriv_knots, i, included[j], included[j+1])
            i += 1

        return value

    def _area_integral(self, knots, i, t_lo, t_hi):
        beta0 = self._area_d(knots, i, t_hi) - self._area_d(knots, i, t_lo)
        beta1 = (self._area_b(knots, i, t_hi) + self._area_c(knots, i, t_hi)) - (self._area_b(knots, i, t_lo) + self._area_c(knots, i, t_lo))
        beta2 = self._area_a(knots, i, t_hi) - self._area_a(knots, i, t_lo)

        points = np.concatenate([
            [0.0] * 2,
            tf.nn.softmax(self.logits).numpy(),
            [0.0] * 2
        ])
        coefs = np.array([beta0, beta1, beta2])

        return (points[i-2:i+1] * coefs).sum()

    def _area_a(self, knots, i, t):
        denom = (knots[i+2] - knots[i]) * (knots[i+2] - knots[i+1])
        if denom == 0.0:
            return 0.0
        else:
            return (t**3/3 - t**2 * knots[i] + t * (knots[i])**2) / denom

    def _area_b(self, knots, i, t):
        denom =  (knots[i+1] - knots[i-1]) * (knots[i+1] - knots[i])
        if denom == 0.0:
            return 0.0
        else:
            return (-t**3/3 + t**2/2 * (knots[i-1] + knots[i+1]) - t * knots[i-1] * knots[i+1]) / denom

    def _area_c(self, knots, i, t):
        denom = (knots[i+2] - knots[i]) * (knots[i+1] - knots[i])
        if denom == 0.0:
            return 0.0
        else:
            return (-t**3/3 + t**2/2 * (knots[i] + knots[i+2]) - t * knots[i] * knots[i+2]) / denom

    def _area_d(self, knots, i, t):
        denom =  (knots[i+1] - knots[i-1]) * (knots[i+1] - knots[i])
        if denom == 0.0:
            return 0.0
        else:
            return (t**3/3 - t**2 * knots[i+1] + t * (knots[i+1])**2) / denom

    def expected_value(self, t_lo=0.0, t_hi=1.0):
        assert t_lo >= 0.0 and t_hi <= 1.0

        deriv_knots = self.dist.deriv_knots.numpy()
        included = deriv_knots[np.where((deriv_knots > t_lo) & (deriv_knots < t_hi))].tolist()
        included = [t_lo, *included, t_hi]

        i = np.searchsorted(deriv_knots, t_lo, side='right') - 1

        value = 0.0
        for j in range(len(included) - 1):
            value += self._expected_value_integral(deriv_knots, i, included[j], included[j+1])
            i += 1

        # This works as long as you scale by the area inside the curve
        return value / self.area()

    def _expected_value_integral(self, knots, i, t_lo, t_hi):
        beta0 = self._d(knots, i, t_hi) - self._d(knots, i, t_lo)
        beta1 = (self._b(knots, i, t_hi) + self._c(knots, i, t_hi)) - (self._b(knots, i, t_lo) + self._c(knots, i, t_lo))
        beta2 = self._a(knots, i, t_hi) - self._a(knots, i, t_lo)

        points = np.concatenate([
            [0.0] * 2,
            tf.nn.softmax(self.logits).numpy(),
            [0.0] * 2
        ])
        coefs = np.array([beta0, beta1, beta2])

        return (points[i-2:i+1] * coefs).sum()

    def _a(self, knots, i, t):
        denom = (knots[i+2] - knots[i]) * (knots[i+2] - knots[i+1])
        if denom == 0.0:
            return 0.0
        else:
            return (t**4/4 - 2/3*t**3 * knots[i] + t**2/2 * (knots[i])**2) / denom

    def _b(self, knots, i, t):
        denom =  (knots[i+1] - knots[i-1]) * (knots[i+1] - knots[i])
        if denom == 0.0:
            return 0.0
        else:
            return (-t**4/4 + t**3/3 * (knots[i-1] + knots[i+1]) - t**2/2 * knots[i-1] * knots[i+1]) / denom

    def _c(self, knots, i, t):
        denom = (knots[i+2] - knots[i]) * (knots[i+1] - knots[i])
        if denom == 0.0:
            return 0.0
        else:
            return (-t**4/4 + t**3/3 * (knots[i] + knots[i+2]) - t**2/2 * knots[i] * knots[i+2]) / denom

    def _d(self, knots, i, t):
        denom =  (knots[i+1] - knots[i-1]) * (knots[i+1] - knots[i])
        if denom == 0.0:
            return 0.0
        else:
            return (t**4/4 - 2/3*t**3 * knots[i+1] + t**2/2 * (knots[i+1])**2) / denom

    def get_pdf_points(self):
        return np.concatenate([[0.0], tf.nn.softmax(self.logits).numpy(), [0.0]])

    def get_cdf_points(self):
        return np.concatenate([[0.0], tf.nn.softmax(self.logits).numpy().cumsum()])
