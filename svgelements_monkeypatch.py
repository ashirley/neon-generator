from svgelements import *
import numpy

#######################
# svgelements monkeypatch
# taken from https://github.com/mathandy/svgpathtools/blob/master/svgpathtools/path.py

def line_unit_tangent(self, t=None):
    """returns the unit tangent of the segment at t."""
    assert self.end != self.start
    dseg = self.end - self.start
    return dseg * (1 / abs(dseg))


Line.unit_tangent = line_unit_tangent


def line_normal(self, t=None):
    """returns the (right hand rule) unit normal vector to self at t."""
    return -1j * self.unit_tangent(t)


Line.normal = line_normal


def bezier_unit_tangent(seg, t):
    """Returns the unit tangent of the segment at t.
    Notes
    -----
    If you receive a RuntimeWarning, try the following:
    >>> import numpy
    >>> old_numpy_error_settings = numpy.seterr(invalid='raise')
    This can be undone with:
    >>> numpy.seterr(**old_numpy_error_settings)
    """

    dseg = seg.derivative(t)

    # Note: dseg might be numpy value, use numpy.seterr(invalid='raise')
    try:
        unit_tangent = dseg * (1 / abs(dseg))
    except (ZeroDivisionError, FloatingPointError):
        # TODO

        # This may be a removable singularity, if so we just need to compute
        # the limit.
        # Note: limit{{dseg / abs(dseg)} = sqrt(limit{dseg**2 / abs(dseg)**2})
        dseg_poly = seg.poly().deriv()
        dseg_abs_squared_poly = (real(dseg_poly) ** 2 +
                                 imag(dseg_poly) ** 2)
        try:
            unit_tangent = sqrt(rational_limit(dseg_poly ** 2,
                                               dseg_abs_squared_poly, t))
        except ValueError:
            bef = seg.poly().deriv()(t - 1e-4)
            aft = seg.poly().deriv()(t + 1e-4)
            mes = ("Unit tangent appears to not be well-defined at "
                   "t = {}, \n".format(t) +
                   "seg.poly().deriv()(t - 1e-4) = {}\n".format(bef) +
                   "seg.poly().deriv()(t + 1e-4) = {}".format(aft))
            raise ValueError(mes)
    return unit_tangent


def rational_limit(f, g, t0):
    """Computes the limit of the rational function (f/g)(t)
    as t approaches t0."""
    assert isinstance(f, numpy.poly1d) and isinstance(g, numpy.poly1d)
    assert g != numpy.poly1d([0])
    if g(t0) != 0:
        return f(t0) / g(t0)
    elif f(t0) == 0:
        return rational_limit(f.deriv(), g.deriv(), t0)
    else:
        raise ValueError("Limit does not exist.")


def real(z):
    try:
        return numpy.poly1d(z.coeffs.real)
    except AttributeError:
        return z.real


def imag(z):
    try:
        return numpy.poly1d(z.coeffs.imag)
    except AttributeError:
        return z.imag


def quad_unit_tangent(self, t):
    """returns the unit tangent vector of the segment at t (centered at
    the origin and expressed as a complex number).  If the tangent
    vector's magnitude is zero, this method will find the limit of
    self.derivative(tau)/abs(self.derivative(tau)) as tau approaches t."""
    return bezier_unit_tangent(self, t)


QuadraticBezier.unit_tangent = quad_unit_tangent
QuadraticBezier.normal = line_normal


def quad_derivative(self, t, n=1):
    """returns the nth derivative of the segment at t.
    Note: Bezier curves can have points where their derivative vanishes.
    If you are interested in the tangent direction, use the unit_tangent()
    method instead."""
    p = (self.start, self.control, self.end)
    if n == 1:
        return 2 * ((p[1] - p[0]) * (1 - t) + (p[2] - p[1]) * t)
    elif n == 2:
        return 2 * (p[2] - 2 * p[1] + p[0])
    elif n > 2:
        return 0
    else:
        raise ValueError("n should be a positive integer.")


QuadraticBezier.derivative = quad_derivative


def quad_poly(self, return_coeffs=False):
    """returns the quadratic as a Polynomial object."""
    p = (complex(self.start), complex(self.control), complex(self.end))
    coeffs = (p[0] - 2 * p[1] + p[2], 2 * (p[1] - p[0]), p[0])
    if return_coeffs:
        return coeffs
    else:
        return numpy.poly1d(coeffs)


QuadraticBezier.poly = quad_poly

CubicBezier.unit_tangent = quad_unit_tangent
CubicBezier.normal = line_normal


def cubic_derivative(self, t, n=1):
    """returns the nth derivative of the segment at t.
    Note: Bezier curves can have points where their derivative vanishes.
    If you are interested in the tangent direction, use the unit_tangent()
    method instead."""
    p = (self.start, self.control1, self.control2, self.end)
    if n == 1:
        return 3 * (p[1] - p[0]) * (1 - t) ** 2 + 6 * (p[2] - p[1]) * (1 - t) * t + 3 * (
                p[3] - p[2]) * t ** 2
    elif n == 2:
        return 6 * (
                (1 - t) * (p[2] - 2 * p[1] + p[0]) + t * (p[3] - 2 * p[2] + p[1]))
    elif n == 3:
        return 6 * (p[3] - 3 * (p[2] - p[1]) - p[0])
    elif n > 3:
        return 0
    else:
        raise ValueError("n should be a positive integer.")


CubicBezier.derivative = cubic_derivative


def cubic_poly(self, return_coeffs=False):
    """Returns a the cubic as a Polynomial object."""
    p = (complex(self.start), complex(self.control1), complex(self.control2), complex(self.end))
    coeffs = (-1 * p[0] + 3 * (p[1] - p[2]) + p[3],
              3 * (p[0] - 2 * p[1] + p[2]),
              3 * (-1 * p[0] + p[1]),
              p[0])
    if return_coeffs:
        return coeffs
    else:
        return numpy.poly1d(coeffs)


CubicBezier.poly = cubic_poly

Arc.unit_tangent = quad_unit_tangent
Arc.normal = line_normal


def arc_derivative(self, t, n=1):
    """returns the nth derivative of the segment at t."""
    angle = radians(self.theta + t * self.delta)
    phi = self.get_rotation().as_radians
    rx = self.rx
    ry = self.ry
    k = (self.delta * numpy.pi / 180) ** n  # ((d/dt)angle)**n

    if n % 4 == 0 and n > 0:
        return rx * cos(phi) * cos(angle) - ry * sin(phi) * sin(angle) + 1j * (
                rx * sin(phi) * cos(angle) + ry * cos(phi) * sin(angle))
    elif n % 4 == 1:
        return k * (-rx * cos(phi) * sin(angle) - ry * sin(phi) * cos(angle) + 1j * (
                -rx * sin(phi) * sin(angle) + ry * cos(phi) * cos(angle)))
    elif n % 4 == 2:
        return k * (-rx * cos(phi) * cos(angle) + ry * sin(phi) * sin(angle) + 1j * (
                -rx * sin(phi) * cos(angle) - ry * cos(phi) * sin(angle)))
    elif n % 4 == 3:
        return k * (rx * cos(phi) * sin(angle) + ry * sin(phi) * cos(angle) + 1j * (
                rx * sin(phi) * sin(angle) - ry * cos(phi) * cos(angle)))
    else:
        raise ValueError("n should be a positive integer.")


Arc.derivative = arc_derivative
