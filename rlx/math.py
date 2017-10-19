import numpy as np
import pandas as pd

def angle_vector(v):
    """
        compute angle with respect to the positive X direction
        of the vector given as argument (assumed a vector to the origin)

    :param v: a 2D vector
    :return: the angle in degrees
    """
    # asserts removed for efficiency
    #    assert type(b) == np.ndarray, "vector must be numpy array"
    #    assert b.shape == (2,), "must have vector of length two"
    # implementation for two vectors
    #
    #    return 180 + np.arctan2(a[1] - b[1], a[0] - b[0]) * 180 / np.pi

    return 180 + np.arctan2(-v[1], -v[0]) * 180 / np.pi


