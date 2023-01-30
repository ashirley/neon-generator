from svgelements import *
import numpy
from stl import Mesh

# file = "examples/lines.svg"
# file = "examples/freehand.svg"
# file = "examples/freehand-simpler.svg"
# file = "examples/curves.svg"
file = "examples/tree.svg"
neon_length = "500cm"
res = 50  # How man steps we split a curve into. Higher the better quality but bigger STL


def stuff():
    svg = SVG.parse(file)
    paths = list(svg.select(lambda e: isinstance(e, Path)))
    stroke_colors = {path.stroke.hex for path in paths}

    if len(paths) == 0:
        print("ERROR: no paths")
        return 1

    if len(stroke_colors) != 1:
        print("ERROR: multiple color paths")
        return 1

    total_length = sum(path.length() for path in paths)
    scale = Length(neon_length).value(ppi=DEFAULT_PPI) / total_length

    print("scale: %s" % scale)

    # TODO: check for sharp angles
    # TODO: check for overlapping
    # TODO: adjust more than just the last line to meet the next segment.
    #  Especially if it is a short segment, resolution is high or there is a large difference in angle needed.

    # walk the path, generating the quad-2d coordinates for:
    # * outer left wall
    # * inner left wall
    # * inner right wall
    # * outer right wall

    quads = []

    for path in paths:

        prev_join = None  # angle between previous segment and curr
        for i in range(len(path)):
            curr_segment = path[i] * ("scale(%d)" % scale)  # current segment
            next_segment = path[i + 1] * ("scale(%d)" % scale) if i + 1 < len(path) else None  # next segment, if any
            curr_join = None  # angle between the current segment and the next

            if isinstance(curr_segment, Move):
                # A move doesn't need neon or a support
                print("Move: %s" % curr_segment)
            elif isinstance(curr_segment, Close):
                # TODO
                print("Close: %s" % curr_segment)
            elif isinstance(curr_segment, Line):
                print("Line: %s" % curr_segment)
                # Straight lines only need 2 quad-2d points along the normal at the start and end 
                # (although we only put the end in if the next segment won't duplicate it).
                curr_join = join_angle(next_segment, curr_segment.normal())

                if curr_join is None:
                    # not joining the next segment, add both a start and an end.
                    quads.extend([
                        quad_point(curr_segment.start, prev_join, curr_segment.normal()),
                        quad_point(curr_segment.end, curr_join, curr_segment.normal())
                    ])
                else:
                    # joining the next segment, just add a start and leave the next segment to do our end (it's start)
                    quads.append(quad_point(curr_segment.start, prev_join, curr_segment.normal()))

            elif isinstance(curr_segment, Arc):
                print("Arc: %s" % curr_segment)
                curr_join = join_angle(next_segment, curr_segment.normal(0.99))

                quads.append(quad_point(curr_segment.point(0), prev_join, curr_segment.normal(0)))
                for j in range(1, res):
                    quads.append(quad_point(curr_segment.point(j / res), None, curr_segment.normal(j / res)))

                if curr_join is None:
                    # not joining the next segment, add an end along the normal of the curve
                    quads.append(quad_point(curr_segment.point(0.99), curr_join, curr_segment.normal(0.99)))

            elif isinstance(curr_segment, QuadraticBezier):
                print("QuadraticBezier: %s" % curr_segment)
                curr_join = join_angle(next_segment, curr_segment.normal(0.99))

                quads.append(quad_point(curr_segment.point(0), prev_join, curr_segment.normal(0)))
                for j in range(1, res):
                    quads.append(quad_point(curr_segment.point(j / res), None, curr_segment.normal(j / res)))

                if curr_join is None:
                    # not joining the next segment, add an end along the normal of the curve
                    quads.append(quad_point(curr_segment.point(0.99), curr_join, curr_segment.normal(0.99)))

            elif isinstance(curr_segment, CubicBezier):
                print("CubicBezier: %s" % curr_segment)
                curr_join = join_angle(next_segment, curr_segment.normal(0.99))

                quads.append(quad_point(curr_segment.point(0), prev_join, curr_segment.normal(0)))
                for j in range(1, res):
                    quads.append(quad_point(curr_segment.point(j / res), None, curr_segment.normal(j / res)))

                if curr_join is None:
                    # not joining the next segment, add an end along the normal of the curve
                    quads.append(quad_point(curr_segment.point(0.99), curr_join, curr_segment.normal(0.99)))

            prev_join = curr_join

    for quad in quads:
        print(quad)

    # TODO: this assumes quads is a single contiguous, non-closed shape.

    # Each quad is 8 vertices (2 z positions for each 4 2d points)
    # between each set of 8 vertices is 8 square faces = 16 triangles)
    # At the end is 6 triangles for the 1 face.

    # Build the stl triangles.
    stl_data = numpy.zeros((len(quads) - 1) * 16 + 12, dtype=Mesh.dtype)
    stl_mesh = Mesh(stl_data, remove_empty_areas=False)

    #  ---    ---    wall_width + channel_depth
    # |0\1|__|4/ |   wall_width
    # |  / \  \ 5|
    # | / 2  \3\ |
    # |/_______\\|   0
    #
    # 0   1  2   3

    stl_mesh.vectors[0] = numpy.array([
        [quads[0][0].x, quads[0][0].y, wall_width + channel_depth],
        [quads[0][1].x, quads[0][1].y, wall_width],
        [quads[0][0].x, quads[0][0].y, 0],
    ])

    stl_mesh.vectors[1] = numpy.array([
        [quads[0][0].x, quads[0][0].y, wall_width + channel_depth],
        [quads[0][1].x, quads[0][1].y, wall_width + channel_depth],
        [quads[0][1].x, quads[0][1].y, wall_width],
    ])

    stl_mesh.vectors[2] = numpy.array([
        [quads[0][1].x, quads[0][1].y, wall_width],
        [quads[0][3].x, quads[0][3].y, 0],
        [quads[0][0].x, quads[0][0].y, 0],
    ])

    stl_mesh.vectors[3] = numpy.array([
        [quads[0][1].x, quads[0][1].y, wall_width],
        [quads[0][2].x, quads[0][2].y, wall_width],
        [quads[0][3].x, quads[0][3].y, 0],
    ])

    stl_mesh.vectors[4] = numpy.array([
        [quads[0][2].x, quads[0][2].y, wall_width + channel_depth],
        [quads[0][3].x, quads[0][3].y, wall_width + channel_depth],
        [quads[0][2].x, quads[0][2].y, wall_width],
    ])

    stl_mesh.vectors[5] = numpy.array([
        [quads[0][2].x, quads[0][2].y, wall_width],
        [quads[0][3].x, quads[0][3].y, wall_width + channel_depth],
        [quads[0][3].x, quads[0][3].y, 0],
    ])

    for i in range(1, len(quads)):
        prev_quads = quads[i - 1]
        curr_quads = quads[i]

        v = (i - 1) * 16 + 6

        stl_mesh.vectors[v] = numpy.array([
            [prev_quads[0].x, prev_quads[0].y, 0],
            [curr_quads[0].x, curr_quads[0].y, 0],
            [prev_quads[0].x, prev_quads[0].y, wall_width + channel_depth],
        ])

        stl_mesh.vectors[v + 1] = numpy.array([
            [prev_quads[0].x, prev_quads[0].y, wall_width + channel_depth],
            [curr_quads[0].x, curr_quads[0].y, 0],
            [curr_quads[0].x, curr_quads[0].y, wall_width + channel_depth],
        ])

        stl_mesh.vectors[v + 2] = numpy.array([
            [prev_quads[0].x, prev_quads[0].y, wall_width + channel_depth],
            [curr_quads[0].x, curr_quads[0].y, wall_width + channel_depth],
            [curr_quads[1].x, curr_quads[1].y, wall_width + channel_depth],
        ])

        stl_mesh.vectors[v + 3] = numpy.array([
            [prev_quads[0].x, prev_quads[0].y, wall_width + channel_depth],
            [curr_quads[1].x, curr_quads[1].y, wall_width + channel_depth],
            [prev_quads[1].x, prev_quads[1].y, wall_width + channel_depth],
        ])

        stl_mesh.vectors[v + 4] = numpy.array([
            [prev_quads[1].x, prev_quads[1].y, wall_width + channel_depth],
            [curr_quads[1].x, curr_quads[1].y, wall_width + channel_depth],
            [prev_quads[1].x, prev_quads[1].y, wall_width],
        ])

        stl_mesh.vectors[v + 5] = numpy.array([
            [prev_quads[1].x, prev_quads[1].y, wall_width],
            [curr_quads[1].x, curr_quads[1].y, wall_width + channel_depth],
            [curr_quads[1].x, curr_quads[1].y, wall_width],
        ])

        stl_mesh.vectors[v + 6] = numpy.array([
            [prev_quads[1].x, prev_quads[1].y, wall_width],
            [curr_quads[1].x, curr_quads[1].y, wall_width],
            [prev_quads[2].x, prev_quads[2].y, wall_width],
        ])

        stl_mesh.vectors[v + 7] = numpy.array([
            [prev_quads[2].x, prev_quads[2].y, wall_width],
            [curr_quads[1].x, curr_quads[1].y, wall_width],
            [curr_quads[2].x, curr_quads[2].y, wall_width],
        ])

        stl_mesh.vectors[v + 8] = numpy.array([
            [prev_quads[2].x, prev_quads[2].y, wall_width],
            [curr_quads[2].x, curr_quads[2].y, wall_width],
            [curr_quads[2].x, curr_quads[2].y, wall_width + channel_depth],
        ])

        stl_mesh.vectors[v + 9] = numpy.array([
            [prev_quads[2].x, prev_quads[2].y, wall_width],
            [curr_quads[2].x, curr_quads[2].y, wall_width + channel_depth],
            [prev_quads[2].x, prev_quads[2].y, wall_width + channel_depth],
        ])

        stl_mesh.vectors[v + 10] = numpy.array([
            [prev_quads[2].x, prev_quads[2].y, wall_width + channel_depth],
            [curr_quads[2].x, curr_quads[2].y, wall_width + channel_depth],
            [prev_quads[3].x, prev_quads[3].y, wall_width + channel_depth],
        ])

        stl_mesh.vectors[v + 11] = numpy.array([
            [prev_quads[3].x, prev_quads[3].y, wall_width + channel_depth],
            [curr_quads[2].x, curr_quads[2].y, wall_width + channel_depth],
            [curr_quads[3].x, curr_quads[3].y, wall_width + channel_depth],
        ])

        stl_mesh.vectors[v + 12] = numpy.array([
            [prev_quads[3].x, prev_quads[3].y, wall_width + channel_depth],
            [curr_quads[3].x, curr_quads[3].y, wall_width + channel_depth],
            [prev_quads[3].x, prev_quads[3].y, 0],
        ])

        stl_mesh.vectors[v + 13] = numpy.array([
            [prev_quads[3].x, prev_quads[3].y, 0],
            [curr_quads[3].x, curr_quads[3].y, wall_width + channel_depth],
            [curr_quads[3].x, curr_quads[3].y, 0],
        ])

        stl_mesh.vectors[v + 14] = numpy.array([
            [prev_quads[3].x, prev_quads[3].y, 0],
            [curr_quads[3].x, curr_quads[3].y, 0],
            [prev_quads[0].x, prev_quads[0].y, 0],
        ])

        stl_mesh.vectors[v + 15] = numpy.array([
            [curr_quads[3].x, curr_quads[3].y, 0],
            [curr_quads[0].x, curr_quads[0].y, 0],
            [prev_quads[0].x, prev_quads[0].y, 0],
        ])

    #  ---    ---    wall_width + channel_depth
    # |0\1|__|4/ |   wall_width
    # |  / \  \ 5|
    # | / 2  \3\ |
    # |/_______\\|   0
    #
    # 3   2  1   0

    i = len(quads) - 1
    v = i * 16 + 6

    stl_mesh.vectors[v] = numpy.array([
        [quads[i][3].x, quads[i][3].y, wall_width + channel_depth],
        [quads[i][2].x, quads[i][2].y, wall_width],
        [quads[i][3].x, quads[i][3].y, 0],
    ])

    stl_mesh.vectors[v + 1] = numpy.array([
        [quads[i][3].x, quads[i][3].y, wall_width + channel_depth],
        [quads[i][2].x, quads[i][2].y, wall_width + channel_depth],
        [quads[i][2].x, quads[i][2].y, wall_width],
    ])

    stl_mesh.vectors[v + 2] = numpy.array([
        [quads[i][2].x, quads[i][2].y, wall_width],
        [quads[i][0].x, quads[i][0].y, 0],
        [quads[i][3].x, quads[i][3].y, 0],
    ])

    stl_mesh.vectors[v + 3] = numpy.array([
        [quads[i][2].x, quads[i][2].y, wall_width],
        [quads[i][1].x, quads[i][1].y, wall_width],
        [quads[i][0].x, quads[i][0].y, 0],
    ])

    stl_mesh.vectors[v + 4] = numpy.array([
        [quads[i][1].x, quads[i][1].y, wall_width + channel_depth],
        [quads[i][0].x, quads[i][0].y, wall_width + channel_depth],
        [quads[i][1].x, quads[i][1].y, wall_width],
    ])

    stl_mesh.vectors[v + 5] = numpy.array([
        [quads[i][1].x, quads[i][1].y, wall_width],
        [quads[i][0].x, quads[i][0].y, wall_width + channel_depth],
        [quads[i][0].x, quads[i][0].y, 0],
    ])

    print("Saving %d triangles..." % len(stl_mesh.vectors))
    stl_mesh.save("output.stl")


channel_width = 10
channel_depth = 15
wall_width = 2


# generate 4 points along the join line, centered on the given point with the given normal.
def quad_point(center, join, normal):
    # find the angle between the join line and the normal
    # 1/cos(angle) is how much bigger the distances should be between center and the quad points
    if join is None:
        join = normal

    angle = Point(0).angle_to(normal).as_radians - Point(0).angle_to(join).as_radians
    multiple = 1 / cos(angle)

    # print ("creating point from center: (%s), join: (%s), normal: (%s), angle: %s (%s), multiplier: %s" % (str(center), str(join), str(abs(join)), str(normal), str(angle), str(multiple)))

    a = join * multiple * (channel_width / 2)
    b = join * multiple * (channel_width / 2 + wall_width)
    return center + b, center + a, center - a, center - b


def join_angle(next_segment, curr_normal):
    if next_segment is None:
        return None
    elif isinstance(next_segment, Move):
        # A move doesn't need neon or a support
        return None
    elif isinstance(next_segment, Close):
        # TODO
        return None
    elif isinstance(next_segment, Line):
        # join is a unit vector between the 2 normals
        curr_join = curr_normal + next_segment.normal()
    elif isinstance(next_segment, Arc):
        curr_join = curr_normal + next_segment.normal(0)
    elif isinstance(next_segment, QuadraticBezier):
        curr_join = curr_normal + next_segment.normal(0)
    elif isinstance(next_segment, CubicBezier):
        curr_join = curr_normal + next_segment.normal(0)

    # make a unit vector
    return curr_join * (1 / abs(curr_join))


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

if __name__ == "__main__":
    stuff()
