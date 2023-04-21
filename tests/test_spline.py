import numpy as np

from spline_filter.main import lerp, bezier_curve, cubic_bezier_curve, Spline

def test_lerp():
    p1, p2 = np.array([0., 0.]), np.array([1., 1.])
    u = np.linspace(0., 1., 1000)
    # Interpolated points
    interp = []
    # Expected points
    expected = [np.array([val, val]) for val in u]

    for val in u:
        interp.append(lerp(p1, p2, val))
    
    interp = np.array(interp)
    assert np.isclose(interp, expected).all()

def test_bezier():
    p1, p2, p3 = np.array([0., 0.]), np.array([0.5, 0.5]), np.array([1., 1.])
    u = np.linspace(0., 1., 1000)
    # Interpolated points
    interp = []
    # Expected points
    expected = [np.array([val, val]) for val in u]

    for val in u:
        interp.append(bezier_curve(p1, p2, p3, val))
    
    interp = np.array(interp)
    assert np.isclose(interp, expected).all()

def test_cubic_bezier():
    p1, p2, p3, p4 = np.array([0., 0.]), np.array([1./3., 1./3.]), np.array([2./3., 2./3.]), np.array([1., 1.])
    u = np.linspace(0., 1., 1000)
    # Interpolated points
    interp = []
    # Expected points
    expected = [np.array([val, val]) for val in u]

    for val in u:
        interp.append(cubic_bezier_curve(p1, p2, p3, p4, val))
    
    interp = np.array(interp)
    assert np.isclose(interp, expected).all()

def test_spline():
    assert False, "WIP"