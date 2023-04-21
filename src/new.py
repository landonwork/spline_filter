'''Copyright © 2023 Landon Work. All rights reserved.'''

from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

ArrayLike = Union[tf.Tensor, np.ndarray]

def lerp(p1, p2, u):
    # assert is_point(p1) and is_point(p2), f'{p1}, {p2}'
    assert 0 <= u and u <= 1, 'invalid `u`'
    return p1 * (1-u) + p2 * u

def bezier_curve(p1, p2, p3, u):
    return lerp(lerp(p1, p2, u), lerp(p2, p3, u), u)

def cubic_bezier_curve(p1, p2, p3, p4, u = None):
    return lerp(bezier_curve(p1, p2, p3, u), bezier_curve(p2, p3, p4, u), u)

def quadratic_equation_pos(a, b, c):
    # print('a, b, c:', a, b, c)
    # print('Discriminant:', b * b - 4 * a * c)
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

class Spline:
    '''Thin wrapper around an output tensor that provides convenient
    methods for inspecting, visualizing, etc. a spline filter'''
    def __init__(self, control_points: tf.Tensor):
        '''An array-like object containing the control points

        Parameters:
        ===========
        :param control_points: a 2-D floating-point array or tensor with exactly
            two columns and an odd number of rows
        '''
        self.control_points = control_points
        self.n_curves = (control_points.shape[0] - 1) // 2

    def plot(self, step: float = 1e-2):
        # Interpolation
        interp = self.interpolate(step)
        plt.plot(interp[:, 0], interp[:, 1])
        # Control points
        plt.plot(
            self.control_points[:, 0],
            self.control_points[:, 1],
            linestyle='dashed',
            marker='s',
            alpha=0.8
        )

    def point_at_u(self, u: float):
        '''get (x, y) for value `u`

        Parameters:
        ===========
        :param u: A floating-point number between 0 and 1. The boundaries for
            different curves within the spline are multiples of
            (1 / number of curves)
        '''
        if u == self.n_curves:
            return self.control_points[-1,:]
        elif u > self.n_curves:
            raise ValueError('u: {u}')
        else:
            index = int(u) * 2
            val = float(u) % 1.
            return bezier_curve(*[self.control_points[ind, :] for ind in range(index, index+3)], val)

    def interpolate(self, step = 1e-2):
        '''Get an array of points drawn all along the spline from u = 0 to 
        u = 1

        Parameters:
        ===========
        :param step: The step size to use for `u`
        
        Returns:
        ===========
        A numpy array with X and Y coordinates of points along the spline
        with an even U spacing
        '''
        u = np.arange(0., self.n_curves + step, step)
        return np.array([self.point_at_u(val) for val in u])

    def __call__(self, u):
        self.point_at_u(u)

    @classmethod
    def from_vectors(cls, velocities):
        '''Alternate constructor that takes velocities instead of control points
        and creates a C1 spline. The resulting spline filter starts at
        x = 0, y = 0, u = 0 and ends at x = 1, y = 1, u = 1. It is useful for
        constructing flexible CDFs.

        Parameters:
        ===========
        :param velocities: a 2-D array or tensor with 2 columns and any number of rows
        '''
        points = [np.zeros(2)]

        for i in range(velocities.shape[0] - 1):
            v0 = velocities[i, :]
            v1 = velocities[i+1, :]

            points.append(points[-1] + v0)
            points.append(points[-1] + v1)
        
        points = np.array(points)
        points = points / points[-1, :]

        return cls(points)

class SplineFilter(keras.layers.Layer):
    '''Keras-compatible layer that transforms inputs into a spline filter.
    Output from the previous layer must be and 1-D tensor positive and non-zero
    and have an even number of elements that is greater or equal to 4. In
    specific, inputs are treated as velocities that are used to construct the
    control points for a bezier spline that represents a continuous CDF.
    '''
    def __init__(self, min: float = 0, max: float = 1, **kwargs):
        kwargs['trainable'] = False
        super().__init__(**kwargs)
        self.M = None
        # Used to scale X outputs correctly
        self.x_range = max - min
        self.x_min = min

    def build(self, input_shape):
        assert input_shape[0] is None, f'{input_shape}'
        assert len(input_shape) == 2, f'{input_shape}'
        assert input_shape[1] >= 4 and input_shape[1] % 2 == 0, f'{input_shape}'

        output_shape = self.compute_output_shape(input_shape)

        # It might have been easier to do this with a more sparse array and a cumsum within rows
        M = np.zeros(((output_shape[3] + 1) // 2, output_shape[3]), np.float32)
        row = 0
        col = 1

        while col != M.shape[-1]:
            # Copy the previous column
            M[:, col] = M[:, col-1]
            # Only increment the first row once, then move to the next
            if row == 0:
                if M[row, col] == 0 and col == 1:
                    M[row, col] = 1
                    row += 1
                else:
                    assert False
            # Otherwise, increment the next highest row
            else:
                if M[row, col] == 2:
                    row += 1
                M[row, col] += 1
            # Always move to the next column
            col += 1

        self.M = tf.convert_to_tensor(M)
        self.N = input_shape[1] // 2 - 1
        # Trying to avoid accidentally name shadowing something
        self.range_tensor = tf.convert_to_tensor(np.vstack(
            [np.full(output_shape[-1], self.x_range, np.float32), np.ones(output_shape[-1], np.float32)]
        ).reshape((2, 1, -1)))
        self.min_tensor = tf.convert_to_tensor(np.vstack(
            [np.full(output_shape[-1], self.x_min, np.float32), np.zeros(output_shape[-1], np.float32)]
        ).reshape((2, 1, -1)))
    
    def call(self, inputs):
        two_rows = tf.reshape(inputs, [-1, 2, 1, inputs.shape[1] // 2])
        points = tf.matmul(two_rows, self.M)
        # Achieve proper scaling
        scaled = (points / points[:, :, :, -1:]) * self.range_tensor + self.min_tensor

        return scaled

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = tuple(input_shape.as_list())
        return tf.TensorShape([None, 2, 1, input_shape[1] - 1])

@tf.function
def neg_log_spline_density(y_true, y_pred):
    '''Loss function for a spline output layer used as a probability
    distribution

    Parameters:
    ===========
    :param y_true: a 1-D tensor where each element is an observed value for the
        distribution of y
    :param y_pred: a tensor with shape [BATCH_SIZE, 2, 1, N*2+1]
    '''
    # Get the correct curve for each prediction
    y_true, curves = tf.map_fn(get_curve, (y_true, y_pred))
    # Convert the x value to a u value
    u = u_at_x(y_true, curves)
    # Calculate dy/du and dy/dx and divide
    deriv = deriv_at_u(u, curves)
    # Return the negative log transform of the probability density
    return -tf.math.log(deriv)

@tf.function # Supposed to help with optimization
def get_curve(args):
    '''Retrieve the correct bezier curve from a spline given an x value.
    Used in tf.map_fn. I suspect this is where tensor operations would
    be slowest.

    Parameters:
    ===========
    :param args: A tuple containing scalar tensor `x` and tensor `spline`
        which has shape [2, 1, N*2+1]

    Returns:
    ===========
    A tuple with `x` as is and a tensor with shape [2, 1, 3] (a single bezier curve)
    '''
    x, spline = args
    x_bounds = spline[0, 0, 0::2]
    # Get index of the correct curve
    bools = x_bounds <= x
    index = tf.maximum((tf.reduce_sum(tf.cast(bools, tf.int32)) - 1) * 2, 0)

    # return (x, tf.gather(spline, [index, index+1, index+2], axis=2))
    return (x, spline[:, :, index:index+3])

@tf.function
def u_at_x(x, curve):
    '''Get the 'u' for a bezier curve given an 'x'

    Parameters:
    ===========
    :param x: tensor with shape [BATCH_SIZE] representing observed x's
    :param curve: tensor with shape [None, 2, 1, 3] representing bezier curves
    '''
    # And solve for u
    return quadratic_equation_pos(
             curve[:, 0, 0, 0] - 2 * curve[:, 0, 0, 1] + curve[:, 0, 0, 2],
        -2 * curve[:, 0, 0, 0] + 2 * curve[:, 0, 0, 1],
             curve[:, 0, 0, 0] - x
    )

@tf.function
def deriv_at_u(u, curve):
    '''Get the derivative (dy/dx) for a bezier curve at u

    Parameters:
    ===========
    :param u: tensor with shape [BATCH_SIZE] representing observed u's
    :param curve: tensor with shape [None, 2, 1, 3] representing bezier curves

    Returns:
    ===========
    dy_dx as a tensor with shape [BATCH_SIZE]
    '''
    M = tf.constant(
        [[1., -2.,  1.],
         [0.,  2., -2.],
         [0.,  0.,  1.]]
    )

    # [BATCH_SIZE, 1, 3]
    # X = curve[:, 0, :, :]
    # Y = curve[:, 1, :, :]
    # [BATCH_SIZE, 3, 1]
    # Pad the left side of `u` with 1s and then 0s so each row is [0, 1, 2*u]
    u = tf.reshape(u, [-1, 1, 1])
    dU = u + u
    dU = tf.pad(dU, [[0, 0], [1, 0], [0, 0]], constant_values=1)
    dU = tf.pad(dU, [[0, 0], [1, 0], [0, 0]], constant_values=0)
    # Y dot M dot dU / X dot M dot dU
    # [BATCH_SIZE, 1, 3] dot [3, 3] dot [BATCH_SIZE, 3, 1] -> [BATCH_SIZE]
    dy_dx = tf.matmul(tf.matmul(curve[:, 1, :, :], M), dU) / tf.matmul(tf.matmul(curve[:, 0, :, :], M), dU)

    return dy_dx

@tf.function
def quadratic_equation_pos(a, b, c):
    '''The quadratic formula. For some reason, for CDFs at least,
    solving with the positive square root of the discriminant seems to always
    work. Errors occur when the discriminant is zero.
    '''
    return (-b + tf.sqrt(b * b - 4 * a * c)) / (2 * a)
