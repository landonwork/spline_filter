import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
from typing import Union

from spline_filter.spline import BSpline1D
from spline_filter.cdf import BSplineCDF


def tf_basis_batched(knots, degree, t, dtype=tf.float64):
    knots = tf.reshape(knots, [1, -1])
    t = tf.reshape(t, [-1, 1])
    basis = tf.where(
        (knots[:, :-1] <= t) & (t < knots[:, 1:]),
        tf.convert_to_tensor(1.0, dtype=dtype),
        tf.convert_to_tensor(0.0, dtype=dtype),
    )

    num1 = t - knots
    num2 = knots - t
    for r in range(1, degree + 1):
        denom = knots[:, r:] - knots[:, :-r]
        denom = tf.where(denom == 0, tf.convert_to_tensor(np.inf, dtype=dtype), denom)
        basis = (
            num1[:, :-r-1] / denom[:, :-1] * basis[:, :-1]
            + num2[:, r+1:] / denom[:, 1:] * basis[:, 1:]
        )

    return basis


class SimpleDensityEstimator:
    def __init__(self, interior_knots=5, dtype=None):
        """Constructs a SimpleDensityEstimator with randomly initialized control points.
        
        Parameters:
        =============
        :param interior_knots: (int, array-like) - Number of interior knots or
                        values of the interior knots sorted in ascending order. Defaults to 5.
        :param dtype: (np.float32, np.float64) - Numpy data type to use for
                        precision. Must be. If None, defaults to dtype of interior_knots if it
                        is an Numpy ndarray. Otherwise, defaults to np.float64.
        """

        error_msg = f'Argument `interior_knots` must be a sorted 1-D array-like with length greater than 3 or an integer greater than 3. Got {interior_knots}.'
        if isinstance(interior_knots, int):
            if interior_knots <= 3:
                raise ValueError(error_msg)
            self.m = interior_knots
            self.n = self.m - 2  # n = m - degree + 1 = m - 3 + 1
            self.N = self.n + 4  # N = n + 2 * (degree - 1) = n + 2 * (3 - 1)
            self.M = self.n + 8  # M == N + degree + 1 = n + 2 * (degree - 1) + degree + 1 = n + 3 * degree - 1 = n + 3 * 3 - 1
            self.degree = 3
    
            self.dtype = dtype or np.float64
            interior_knots = np.linspace(0.0, 1.0, self.m)
            self._area = tf.convert_to_tensor(1 / (self.m - 1), dtype=self.dtype)  # Area calculation is simple for evenly spaced knots

        elif isinstance(interior_knots, list):
            if not all(isinstance(x, float) for x in interior_knots) or len(interior_knots) <= 3 or sorted(interior_knots) != interior_knots:
                raise ValueError(error_msg)
            if interior_knots[0] != 0.0 or interior_knots[-1] != 1.0:
                raise ValueError('Array-like `interior_knots` must begin with 0.0 and end with 1.0')
                
            self.m = len(interior_knots)
            self.n = self.m - 2  # n = m - degree + 1 = m - 3 + 1
            self.N = self.n + 4  # N = n + 2 * (degree - 1) = n + 2 * (3 - 1)
            self.M = self.n + 8  # M == N + degree + 1 = n + 2 * (degree - 1) + degree + 1 = n + 3 * degree - 1 = n + 3 * 3 - 1
            self.degree = 3
    
            self.dtype = dtype or np.float64
            self._area = None

        elif isinstance(interior_knots, np.ndarray):
            if interior_knots.dtype not in (np.float64, np.float32) or len(interior_knots.shape) != 1 or interior_knots.shape[0] <= 3:
                raise ValueError(error_msg)
            if interior_knots[0] != 0.0 or interior_knots[-1] != 1.0:
                raise ValueError('Array-like `interior_knots` must begin with 0.0 and end with 1.0')

            self.m = interior_knots.shape[0]
            self.n = self.m - 2  # n = m - degree + 1 = m - 3 + 1
            self.N = self.n + 4  # N = n + 2 * (degree - 1) = n + 2 * (3 - 1)
            self.M = self.n + 8  # M == N + degree + 1 = n + 2 * (degree - 1) + degree + 1 = n + 3 * degree - 1 = n + 3 * 3 - 1
            self.degree = 3
    
            self.dtype = dtype or interior_knots.dtype
            self._area = None

        else:
            raise TypeError(error_msg)
            
        if self.n == 2:
            print('Warning: density with n == 2 control points is static')

        self.interior_knots = tf.convert_to_tensor(interior_knots, dtype=self.dtype)
        self.knots = tf.convert_to_tensor(np.concatenate([[0.0] * 3, interior_knots, [1.0] * 3]), dtype=self.dtype)
        self.deriv_knots = tf.convert_to_tensor(np.concatenate([[0.0] * 2, interior_knots, [1.0] * 2]), dtype=self.dtype)

        self.point_logits = tf.Variable(tf.random.uniform([self.n - 1], dtype=self.dtype))
        self.optimizer = tf.keras.optimizers.Adam()

    def neg_log_density(self, t_true, deriv_interior_points):  # t_true should be 1-D, deriv_interior_points should be 2-D
        n_rows = tf.shape(deriv_interior_points)[0]
        zeros = tf.zeros([n_rows, 1], dtype=self.dtype)
        deriv_points = tf.concat(
            [zeros, zeros, deriv_interior_points, zeros, zeros],
            axis=1)
        basis = tf_basis_batched(self.deriv_knots, 2, t_true, dtype=self.dtype)
        density = tf.reduce_sum(deriv_points * basis, axis=1)
        return -tf.math.log(density)

    def neg_log_avg_density(self, t_lo, t_hi, deriv_interior_points):
        """For when we know that the actual value of t is between two points.
        Please make sure that t_lo and t_hi are never exactly 0 or 1.
        """
        n_rows = tf.shape(interiors)[0]
        zeros = tf.zeros([n_rows, 1], dtype=self.dtype)
        ones = tf.ones([n_rows, 1], dtype=self.dtype)
        points = tf.concat(
            [zeros, zeros, zeros, tf.cumsum(deriv_interior_points, axis=-1), ones, ones],
            axis=1)

        basis_lo = tf_basis_batched(self.knots, 3, t_lo, dtype=self.dtype)
        basis_hi = tf_basis_batched(self.knots, 3, t_hi, dtype=self.dtype)
        cdf_lo = tf.reduce_sum(points * basis_lo, axis=1)
        cdf_hi = tf.reduce_sum(points * basis_hi, axis=1)

        # avg_density = (F(t_hi) - F(t_lo)) / (t_hi - t_lo)
        # Properties of logarithms allow us to extract the denominator into a
        # separate log and subtract it from the log of the numerator; and then
        # we distribute the negative which flips the signs; but I'm not sure this
        # is any different from neg_log_cdf because t_hi and t_lo end up in their own
        # additive term and they don't affect the gradient, right?
        loss = -tf.math.log(cdf_hi - cdf_lo) + tf.math.log(t_hi - t_lo)
        return loss

    def neg_log_cdf(self, t_pred, over_pred_vec, deriv_interior_points):
        n_rows = tf.shape(deriv_interior_points)[0]
        zeros = tf.zeros([n_rows, 1], dtype=self.dtype)
        ones = tf.ones([n_rows, 1], dtype=self.dtype)
        points = tf.concat(
            [zeros, zeros, zeros, tf.cumsum(deriv_interior_points, axis=-1), ones, ones],
            axis=1)

        basis = tf_basis_batched(self.knots, 3, t_pred, dtype=self.dtype)
        cdf = tf.reduce_sum(points * basis, axis=1)

        loss = -tf.math.log(tf.where(over_pred_vec > 0.5, cdf, 1 - cdf))
        return loss

    def train(self, t_true, n_epochs=10, batch_size=128, verbose=True):
        for epoch in range(n_epochs):
            if verbose:
                print(f'Epoch {epoch+1}/{n_epochs}', end=' ')
            epoch_loss = 0.0
            for i in range(t_true.shape[0] // batch_size):
                with tf.GradientTape() as tape:
                    deriv_interior_points = tf.reshape(tf.nn.softmax(self.point_logits), [1, -1])
                    loss = self.neg_log_density(
                        t_true[i*batch_size:(i+1)*batch_size],
                        deriv_interior_points)

                epoch_loss += loss.numpy().mean()
                grads = tape.gradient(loss, [self.point_logits])
                self.optimizer.apply_gradients(zip(grads, [self.point_logits]))
            if verbose:
                print(epoch_loss)

    def train_cdf(self, t_pred, over_pred_vec, n_epochs=10, batch_size=128, verbose=True):
        for epoch in range(n_epochs):
            if verbose:
                print(f'Epoch {epoch+1}/{n_epochs}', end=' ')
            epoch_loss = 0.0
            for i in range(t_pred.shape[0] // batch_size):
                with tf.GradientTape() as tape:
                    deriv_interior_points = tf.reshape(tf.nn.softmax(self.point_logits), [1, -1])
                    loss = self.neg_log_cdf(
                        t_pred[i*batch_size:(i+1)*batch_size],
                        over_pred_vec[i*batch_size:(i+1)*batch_size],
                        deriv_interior_points)

                epoch_loss += loss.numpy().mean()
                grads = tape.gradient(loss, [self.point_logits])
                self.optimizer.apply_gradients(zip(grads, [self.point_logits]))
            if verbose:
                print(epoch_loss)

    def area(self):
        # Area changes with the control points if the knots are not evenly
        # spaced; this is unfortunate because we have to make sure that
        # learning does not favor larger areas, but rather treats all the same
        if self._area is None:
            value = tf.convert_to_tensor(0.0, dtype=self.dtype)
            # No gradient required because they are used as integral bounds and the knots are not variable in a SimpleDensityEstimator
            zeros = tf.zeros(2, dtype=self.dtype)
            points = tf.concat(
                [zeros, tf.nn.softmax(self.point_logits), zeros],
                axis=0)
            for i in range(2, self.m + 1):
                value = value + self._area_integral(
                    points,
                    self.deriv_knots,
                    i,
                    self.deriv_knots[i],
                    self.deriv_knots[i+1])

            return value

        # But if they are evenly spaced, area is constant. Knot spacing is decided at the time of construction.
        else:
            return self._area

    def _area_integral(self, points, knots, i, t_lo, t_hi):
        beta0 = self._area_d(knots, i, t_hi) - self._area_d(knots, i, t_lo)
        beta1 = (self._area_b(knots, i, t_hi) + self._area_c(knots, i, t_hi)) - (self._area_b(knots, i, t_lo) + self._area_c(knots, i, t_lo))
        beta2 = self._area_a(knots, i, t_hi) - self._area_a(knots, i, t_lo)
        # tf.stack for stacking scalars rather than tf.concat
        coefs = tf.stack([beta0, beta1, beta2])

        return tf.reduce_sum(points[i-2:i+1] * coefs)

    def _area_a(self, knots, i, t):
        denom = (knots[i+2] - knots[i]) * (knots[i+2] - knots[i+1])
        return tf.where(
            denom == 0.0,
            0.0,
            (t**3/3 - t**2 * knots[i] + t * (knots[i])**2) / denom )

    def _area_b(self, knots, i, t):
        denom = (knots[i+1] - knots[i-1]) * (knots[i+1] - knots[i])
        return tf.where(
            denom == 0.0,
            0.0,
            (-t**3/3 + t**2 * (knots[i-1] + knots[i+1]) / 2 - t * knots[i-1] * knots[i+1]) / denom)

    def _area_c(self, knots, i, t):
        denom = (knots[i+2] - knots[i]) * (knots[i+1] - knots[i])
        return tf.where(
            denom == 0.0,
            0.0,
            (-t**3/3 + t**2 * (knots[i] + knots[i+2]) / 2 - t * knots[i] * knots[i+2]) / denom)

    def _area_d(self, knots, i, t):
        denom = (knots[i+1] - knots[i-1]) * (knots[i+1] - knots[i])
        return tf.where(
            denom == 0.0,
            0.0,
            (t**3/3 - t**2 * knots[i+1] + t * (knots[i+1])**2) / denom)

    def expected_value(self, t_lo=0.0, t_hi=1.0):
        assert t_lo >= 0.0 and t_hi <= 1.0

        deriv_knots = self.deriv_knots.numpy()
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
            tf.nn.softmax(self.point_logits).numpy(),
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
        return np.concatenate([[0.0], tf.nn.softmax(self.point_logits).numpy(), [0.0]])

    def get_cdf_points(self):
        return np.concatenate([[0.0], tf.nn.softmax(self.point_logits).numpy().cumsum()])

    def get_interior_knots(self):
        return self.interior_knots.numpy()

    def get_spline(self):
        return BSpline1D.clamped(self.get_pdf_points(), self.interior_knots.numpy(), 2)

    def get_cdf_spline(self):
        return BSplineCDF.clamped(self.get_cdf_points(), self.interior_knots.numpy(), 3)


class FlexibleDensityEstimator:
    def __init__(self, n=5, dtype=tf.float64):
        # number of interior points in the CDF spline
        assert isinstance(n, int) and n > 1
        if n == 2:
            print('Warning: density with n == 2 is static')

        self.n = n
        self.m = n + 2  # m = n + degree - 1 = n + 3 - 1
        self.N = n + 4  # N = n + 2 * (degree - 1) = n + 2 * (3 - 1)
        self.M = n + 8  # M == N + degree + 1 = n + 2 * (degree - 1) + degree + 1 = n + 3 * degree - 1 = n + 3 * 3 - 1
        self.degree = 3
        self.dtype = dtype

        self.knot_logits = tf.Variable(tf.random.uniform([self.m - 1], dtype=self.dtype))
        self.point_logits = tf.Variable(tf.random.uniform([n - 1], dtype=self.dtype))
        self.optimizer = tf.keras.optimizers.Adam()

    def neg_log_density(self, t_true, interiors): # t_true should be 1-D, deriv_interior_points should be 2-D
        n_rows = tf.shape(interiors)[0]
        zeros = tf.zeros([n_rows, 1], dtype=self.dtype)
        ones = tf.ones([n_rows, 1], dtype=self.dtype)

        deriv_interior_points = tf.reshape(tf.nn.softmax(interiors[:, :self.n-1]), [1, -1])
        deriv_points = tf.concat(
            [zeros, zeros, deriv_interior_points, zeros, zeros],
            axis=1)

        deriv_knots = tf.reshape(
            tf.concat(
                [zeros, zeros, zeros, tf.cumsum(tf.nn.softmax(interiors[:, self.n-1:]), axis=-1), ones, ones],
                axis=1),
            [1, -1])

        basis = tf_basis_batched(deriv_knots, 2, t_true, dtype=self.dtype)
        density = tf.reduce_sum(deriv_points * basis, axis=1)
        # area = self.area() # This is actually very important when the knots can be repositioned but I don't have it implemented with tensors

        # return -tf.math.log(density / area)
        return -tf.math.log(density)

    def neg_log_avg_density(self, t_lo, t_hi, interiors):
        n_rows = tf.shape(interiors)[0]
        zeros = tf.zeros([n_rows, 1], dtype=self.dtype)
        ones = tf.ones([n_rows, 1], dtype=self.dtype)

        deriv_interior_points = tf.reshape(tf.nn.softmax(interiors[:, :self.n-1]), [1, -1])
        points = tf.concat(
            [zeros, zeros, zeros, tf.cumsum(deriv_interior_points, axis=-1), ones, ones],
            axis=1)

        knots = tf.reshape(tf.concat(
            [zeros, zeros, zeros, zeros, tf.cumsum(tf.nn.softmax(interiors[:, self.n-1:]), axis=-1), ones, ones, ones],
            axis=1
        ), [1, -1])

        basis_lo = tf_basis_batched(knots, 3, t_lo, dtype=self.dtype)
        basis_hi = tf_basis_batched(knots, 3, t_hi, dtype=self.dtype)
        cdf_lo = tf.reduce_sum(points * basis_lo, axis=1)
        cdf_hi = tf.reduce_sum(points * basis_hi, axis=1)

        loss = -tf.math.log(cdf_hi - cdf_lo) + tf.math.log(t_hi - t_lo)
        return loss

    def neg_log_cdf(self, t_pred, over_pred_vec, interiors):
        n_rows = tf.shape(interiors)[0]
        zeros = tf.reshape(tf.zeros(n_rows, dtype=self.dtype), [n_rows, 1])
        ones = tf.reshape(tf.ones(n_rows, dtype=self.dtype), [n_rows, 1])

        deriv_interior_points = tf.reshape(tf.nn.softmax(interiors[:, :self.n-1]), [1, -1])
        points = tf.concat(
            [zeros, zeros, zeros, tf.cumsum(deriv_interior_points, axis=-1), ones, ones],
            axis=1)

        knots = tf.reshape(tf.concat(
            [zeros, zeros, zeros, zeros, tf.cumsum(tf.nn.softmax(interiors[:, self.n-1:]), axis=-1), ones, ones, ones],
            axis=1
        ), [1, -1])

        basis = tf_basis_batched(knots, 3, t_pred, dtype=self.dtype)
        cdf = tf.reduce_sum(points * basis, axis=1)

        loss = -tf.math.log(tf.where(over_pred_vec > 0.5, cdf, 1 - cdf))
        return loss

    def train(self, t_true, n_epochs=10, batch_size=1024, verbose=True):
        for epoch in range(n_epochs):
            if verbose:
                print(f'Epoch {epoch+1}/{n_epochs}', end=' ')
            epoch_loss = 0
            for i in range(t_true.shape[0] // batch_size):
                with tf.GradientTape() as tape:
                    loss = self.neg_log_density(
                        t_true[i*batch_size:(i+1)*batch_size],
                        tf.reshape(tf.concat([self.point_logits, self.knot_logits], axis=0), [1, -1])
                    )
                epoch_loss += loss.numpy().mean()
                grads = tape.gradient(loss, [self.point_logits, self.knot_logits])
                self.optimizer.apply_gradients(zip(grads, [self.point_logits, self.knot_logits]))
            if verbose:
                print(epoch_loss)

    def train_cdf(self, t_pred, over_pred_vec, n_epochs=10, batch_size=128, verbose=True):
        for epoch in range(n_epochs):
            if verbose:
                print(f'Epoch {epoch+1}/{n_epochs}', end=' ')
            epoch_loss = 0
            for i in range(t_pred.shape[0] // batch_size):
                with tf.GradientTape() as tape:
                    loss = self.neg_log_cdf(
                        t_pred[i*batch_size:(i+1)*batch_size],
                        over_pred_vec[i*batch_size:(i+1)*batch_size],
                        tf.reshape(tf.concat([self.point_logits, self.knot_logits], axis=0), [1, -1])
                    )
                epoch_loss += loss.numpy().sum()
                grads = tape.gradient(loss, [self.point_logits, self.knot_logits])
                self.optimizer.apply_gradients(zip(grads, [self.point_logits, self.knot_logits]))
            if verbose:
                print(epoch_loss / t_pred.shape[0])  # normalize loss to per observation

    def area(self, t_lo=0.0, t_hi=1.0):
        # assert t_lo >= 0.0 and t_hi <= 1.0
        deriv_knots = np.concatenate([[0.0]*2, self.get_interior_knots(), [1.0]*2])
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
            tf.nn.softmax(self.point_logits).numpy(),
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

        deriv_knots = np.concatenate([[0.0]*2, self.get_interior_knots(), [1.0]*2])
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
            tf.nn.softmax(self.point_logits).numpy(),
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
        return np.concatenate([[0.0], tf.nn.softmax(self.point_logits).numpy(), [0.0]])

    def get_cdf_points(self):
        return np.concatenate([[0.0], tf.nn.softmax(self.point_logits).numpy().cumsum()])

    def get_interior_knots(self):
        arr = np.concatenate([[0.0], tf.nn.softmax(self.knot_logits).numpy().cumsum()])
        arr[-1] = 1.0
        return arr

    def get_spline(self):
        return BSpline1D.clamped(self.get_pdf_points(), self.get_interior_knots(), 2)
