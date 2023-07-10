from svgelements_monkeypatch import *
import svgwrite
import numpy
from stl import Mesh

# file = "examples/lines.svg"
# file = "examples/freehand.svg"
# file = "examples/freehand-simpler.svg"
# file = "examples/curves.svg"
file = "examples/tree.svg"
# file = "examples/2lines.svg"
# file = "examples/2paths.svg"
neon_length = "500cm"
res = 500  # How many steps we split a curve into. Higher the better quality but bigger STL
output_stl_file = "output.stl"
output_debug_svg_file = "output.svg"
log_paths=False
log_quads=False

debug_perpendicular_style = {"stroke":"green", "stroke_width":"1px"}
debug_original_style = {"stroke":"black", "stroke_width":"3px", "fill":"none"}
debug_comb_style = {"stroke":"blue", "stroke_width":"1px"}

def main():
    paths = parse_and_validate_stl()
    scale = 1

    if neon_length:
        total_length = sum(path.length() for path in paths)
        scale = Length(neon_length).value(ppi=DEFAULT_PPI) / total_length

        print("scaling by '%s' to match target length of '%s'" % (scale, neon_length))

    # TODO: check for sharp angles
    # TODO: check for overlapping / fix almost joining paths
    # TODO: adjust more than just the last line to meet the next segment.
    #  Especially if it is a short segment, resolution is high or there is a large difference in angle needed.

    quads_list = generate_perpendicular_quads(paths, scale)

    if output_stl_file:
        # Each quad is 8 vertices (2 z positions for each 4 2d points)
        # between each set of 8 vertices is 8 square faces = 16 triangles)
        # At the end is 6 triangles for the 1 face.

        num_triangles = 0
        for quads in quads_list:
            num_triangles += (len(quads) - 1) * 16 + 12

        # Build the stl triangles.
        stl_data = numpy.zeros(num_triangles, dtype=Mesh.dtype)
        stl_data_cursor = 0

        for quads in quads_list:
            num_triangles_in_quads = (len(quads) - 1) * 16 + 12
            output_array = stl_data['vectors'][stl_data_cursor:stl_data_cursor+num_triangles_in_quads]
            stl_data_cursor += num_triangles_in_quads

            create_stl_triangles(output_array, quads)

        print("Saving %d triangles in %d sections..." % (len(stl_data), len(quads_list)))
        stl_mesh = Mesh(stl_data)
        stl_mesh.save(output_stl_file)

    if output_debug_svg_file:
        dwg = svgwrite.Drawing(output_debug_svg_file, profile='tiny')

        # copy across the original lines
        copy_paths_to_debug_svg(paths, dwg, scale)

        # add the perpendicular lines
        prev_quad = None
        g = dwg.g(**debug_perpendicular_style)
        for quads in quads_list:
            for quad in quads:
                g.add(dwg.line(start=(quad[0][0], quad[0][1]), end=(quad[3][0], quad[3][1])))
                if prev_quad:
                    g.add(dwg.line(start=(prev_quad[0][0], prev_quad[0][1]), end=(quad[0][0], quad[0][1])))
                    g.add(dwg.line(start=(prev_quad[3][0], prev_quad[3][1]), end=(quad[3][0], quad[3][1])))
                prev_quad = quad
            prev_quad = None
        dwg.add(g)

        # add a curvature comb
        add_curvature_comb_to_debug_svg(paths, dwg, scale)

        dwg.save()

    if log_quads:
        for quads in quads_list:
            for quad in quads:
                print(quad)
            print()
            print()

def steps_for_segment(curr_segment, scaled_total_length):
    return max(2, ceil((curr_segment.length() / scaled_total_length) * res))

def parse_and_validate_stl():
    svg = SVG.parse(file)
    paths = list(svg.select(lambda e: isinstance(e, Path)))
    stroke_colors = {path.stroke.hex for path in paths}
    if len(paths) == 0:
        print("ERROR: no paths")
        raise Exception
    if len(stroke_colors) != 1:
        print("ERROR: multiple color paths")
        raise Exception
    return paths


def generate_perpendicular_quads(paths, scale) -> list[list[tuple[Point, Point, Point, Point]]]:
    """
    Walk the path, generating a list of 4 2d coordinates on a line perpendicular to the line of the path.
    The 4 points are for:
     * outer left wall
     * inner left wall
     * inner right wall
     * outer right wall
    When the elements of the path meet, the angle of the "perpendicular" line is altered to ease the transition
    and allow a neat join

    We actually return a list of lists as we split the paths into contiguous sections. For example, if path contains
    2 non-contiguous straight lines, we will return a list of 2 lists, each inner list will have 2 tuples with 4 points
    (one for the start of the line and one for the end)
    """
    scaled_total_length = sum(path.length() for path in paths)

    quads = [[]]
    quad_num = 0;
    for path in paths:

        prev_join = None  # angle between previous segment and curr
        for i in range(len(path)):
            curr_segment = path[i] * ("scale(%d)" % scale)  # current segment
            next_segment = path[i + 1] * ("scale(%d)" % scale) if i + 1 < len(path) else None  # next segment, if any
            curr_join = None  # angle between the current segment and the next

            if isinstance(curr_segment, Move):
                # A move doesn't need neon or a support
                if log_paths:
                    print("Move: %s" % curr_segment)
                if len(quads[quad_num]) > 0:
                    # There is content in the current list, start a new one.
                    quad_num += 1
                    quads.append([])
            elif isinstance(curr_segment, Close):
                # TODO
                if log_paths:
                    print("Close: %s" % curr_segment)
            elif isinstance(curr_segment, Line):
                if log_paths:
                    print("Line: %s" % curr_segment)
                # Straight lines only need 2 quad-2d points along the normal at the start and end
                # (although we only put the end in if the next segment won't duplicate it).
                curr_join = join_angle(next_segment, curr_segment.normal())

                if curr_join is None:
                    # not joining the next segment, add both a start and an end.
                    quads[quad_num].extend([
                        quad_point(curr_segment.start, prev_join, curr_segment.normal()),
                        quad_point(curr_segment.end, curr_join, curr_segment.normal())
                    ])
                else:
                    # joining the next segment, just add a start and leave the next segment to do our end (it's start)
                    quads[quad_num].append(quad_point(curr_segment.start, prev_join, curr_segment.normal()))

            elif isinstance(curr_segment, Arc):
                if log_paths:
                    print("Arc: %s" % curr_segment)
                curr_join = join_angle(next_segment, curr_segment.normal(0.99))

                quads[quad_num].append(quad_point(curr_segment.point(0), prev_join, curr_segment.normal(0)))
                steps = steps_for_segment(curr_segment, scaled_total_length)
                for j in range(1, steps):
                    quads[quad_num].append(quad_point(curr_segment.point(j / steps), None, curr_segment.normal(j / steps)))

                if curr_join is None:
                    # not joining the next segment, add an end along the normal of the curve
                    quads[quad_num].append(quad_point(curr_segment.point(0.99), curr_join, curr_segment.normal(0.99)))

            elif isinstance(curr_segment, QuadraticBezier):
                if log_paths:
                    print("QuadraticBezier: %s" % curr_segment)
                curr_join = join_angle(next_segment, curr_segment.normal(0.99))

                quads[quad_num].append(quad_point(curr_segment.point(0), prev_join, curr_segment.normal(0)))
                steps = steps_for_segment(curr_segment, scaled_total_length)
                for j in range(1, steps):
                    quads[quad_num].append(quad_point(curr_segment.point(j / steps), None, curr_segment.normal(j / steps)))

                if curr_join is None:
                    # not joining the next segment, add an end along the normal of the curve
                    quads[quad_num].append(quad_point(curr_segment.point(0.99), curr_join, curr_segment.normal(0.99)))

            elif isinstance(curr_segment, CubicBezier):
                if log_paths:
                    print("CubicBezier: %s" % curr_segment)
                curr_join = join_angle(next_segment, curr_segment.normal(0.99))

                quads[quad_num].append(quad_point(curr_segment.point(0), prev_join, curr_segment.normal(0)))
                steps = steps_for_segment(curr_segment, scaled_total_length)
                for j in range(1, steps):
                    quads[quad_num].append(quad_point(curr_segment.point(j / steps), None, curr_segment.normal(j / steps)))

                if curr_join is None:
                    # not joining the next segment, add an end along the normal of the curve
                    quads[quad_num].append(quad_point(curr_segment.point(0.99), curr_join, curr_segment.normal(0.99)))

            prev_join = curr_join
    return quads

def copy_paths_to_debug_svg(paths, dwg, scale):
    """
    Copy the paths (as we understand them) out of the source SVG into the debug svg
    """
    minX = float('inf')
    minY = float('inf')
    maxX = 0
    maxY = 0

    for path in paths:
        debug_path = svgwrite.path.Path(**debug_original_style)
        for i in range(len(path)):
            curr_segment = path[i] * ("scale(%d)" % scale)  # current segment
            debug_path.push(curr_segment.d())

        bbox = (path * ("scale(%d)" % scale)).bbox()
        minX = min(minX, bbox[0])
        minY = min(minY, bbox[1])
        maxX = max(maxX, bbox[2])
        maxY = max(maxY, bbox[3])

        dwg.add(debug_path)

    # add a 5% padding
    xPadding = (maxX - minX) * 0.05
    yPadding = (maxY - minY) * 0.05
    minX = minX - xPadding
    maxX = maxX + xPadding
    minY = minY - yPadding
    maxY = maxY + yPadding

    dwg['viewBox'] = "%d %d %d %d" % (minX, minY, maxX - minX, maxY - minY)

def add_curvature_comb_to_debug_svg(paths, dwg, scale):
    """
draw a curvature comb and add it to the debug svg
    """
    scaled_total_length = sum(path.length() for path in paths)
    g = dwg.g(**debug_comb_style)
    for path in paths:
        for i in range(len(path)):
            curr_segment = path[i] * ("scale(%d)" % scale)  # current segment

            if isinstance(curr_segment, Move):
                # A move doesn't need neon or a support so doesn't need a comb either
                pass
            elif isinstance(curr_segment, Close):
                # TODO
                pass
            elif isinstance(curr_segment, Line):
                # Straight lines don't need a comb
                pass
            elif isinstance(curr_segment, Arc):
                prev_end_point = None
                steps = steps_for_segment(curr_segment, scaled_total_length)
                for j in range(0, steps + 1):
                    point = curr_segment.point(j / steps)
                    curvature = curr_segment.curvature(j / steps)
                    normal = curr_segment.normal(j / steps)
                    end_point = point + normal * curvature * 1000

                    g.add(dwg.line(start=point, end=end_point))
                    if prev_end_point:
                        g.add(dwg.line(start=prev_end_point, end=end_point))
                    prev_end_point = end_point

            elif isinstance(curr_segment, QuadraticBezier):
                prev_end_point = None
                steps = steps_for_segment(curr_segment, scaled_total_length)
                for j in range(0, steps + 1):
                    point = curr_segment.point(j / steps)
                    curvature = curr_segment.curvature(j / steps)
                    normal = curr_segment.normal(j / steps)
                    end_point = point + normal * curvature * 1000

                    g.add(dwg.line(start=point, end=end_point))
                    if prev_end_point:
                        g.add(dwg.line(start=prev_end_point, end=end_point))
                    prev_end_point = end_point

            elif isinstance(curr_segment, CubicBezier):
                prev_end_point = None
                steps = steps_for_segment(curr_segment, scaled_total_length)
                for j in range(0, steps + 1):
                    point = curr_segment.point(j / steps)
                    curvature = curr_segment.curvature(j / steps)
                    normal = curr_segment.normal(j / steps)
                    end_point = point + normal * curvature * 1000

                    g.add(dwg.line(start=point, end=end_point))
                    if prev_end_point:
                        g.add(dwg.line(start=prev_end_point, end=end_point))
                    prev_end_point = end_point
    dwg.add(g)



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

    # print ("creating point from center: (%s), join: (%s), normal: (%s), angle: %s (%s), multiplier: %s" % (
    # str(center), str(join), str(abs(join)), str(normal), str(angle), str(multiple)))

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


def create_stl_triangles(output_array, quads):
    """
    Create the triangles for the stl for a single contiguous path.

    """
    #  ---    ---    wall_width + channel_depth
    # |0\1|__|4/ |   wall_width
    # |  / \  \ 5|
    # | / 2  \3\ |
    # |/_______\\|   0
    #
    # 0   1  2   3
    output_array[0] = numpy.array([
        [quads[0][0].x, quads[0][0].y, wall_width + channel_depth],
        [quads[0][1].x, quads[0][1].y, wall_width],
        [quads[0][0].x, quads[0][0].y, 0],
    ])
    output_array[1] = numpy.array([
        [quads[0][0].x, quads[0][0].y, wall_width + channel_depth],
        [quads[0][1].x, quads[0][1].y, wall_width + channel_depth],
        [quads[0][1].x, quads[0][1].y, wall_width],
    ])
    output_array[2] = numpy.array([
        [quads[0][1].x, quads[0][1].y, wall_width],
        [quads[0][3].x, quads[0][3].y, 0],
        [quads[0][0].x, quads[0][0].y, 0],
    ])
    output_array[3] = numpy.array([
        [quads[0][1].x, quads[0][1].y, wall_width],
        [quads[0][2].x, quads[0][2].y, wall_width],
        [quads[0][3].x, quads[0][3].y, 0],
    ])
    output_array[4] = numpy.array([
        [quads[0][2].x, quads[0][2].y, wall_width + channel_depth],
        [quads[0][3].x, quads[0][3].y, wall_width + channel_depth],
        [quads[0][2].x, quads[0][2].y, wall_width],
    ])
    output_array[5] = numpy.array([
        [quads[0][2].x, quads[0][2].y, wall_width],
        [quads[0][3].x, quads[0][3].y, wall_width + channel_depth],
        [quads[0][3].x, quads[0][3].y, 0],
    ])
    for i in range(1, len(quads)):
        prev_quads = quads[i - 1]
        curr_quads = quads[i]

        v = (i - 1) * 16 + 6

        output_array[v] = numpy.array([
            [prev_quads[0].x, prev_quads[0].y, 0],
            [curr_quads[0].x, curr_quads[0].y, 0],
            [prev_quads[0].x, prev_quads[0].y, wall_width + channel_depth],
        ])

        output_array[v + 1] = numpy.array([
            [prev_quads[0].x, prev_quads[0].y, wall_width + channel_depth],
            [curr_quads[0].x, curr_quads[0].y, 0],
            [curr_quads[0].x, curr_quads[0].y, wall_width + channel_depth],
        ])

        output_array[v + 2] = numpy.array([
            [prev_quads[0].x, prev_quads[0].y, wall_width + channel_depth],
            [curr_quads[0].x, curr_quads[0].y, wall_width + channel_depth],
            [curr_quads[1].x, curr_quads[1].y, wall_width + channel_depth],
        ])

        output_array[v + 3] = numpy.array([
            [prev_quads[0].x, prev_quads[0].y, wall_width + channel_depth],
            [curr_quads[1].x, curr_quads[1].y, wall_width + channel_depth],
            [prev_quads[1].x, prev_quads[1].y, wall_width + channel_depth],
        ])

        output_array[v + 4] = numpy.array([
            [prev_quads[1].x, prev_quads[1].y, wall_width + channel_depth],
            [curr_quads[1].x, curr_quads[1].y, wall_width + channel_depth],
            [prev_quads[1].x, prev_quads[1].y, wall_width],
        ])

        output_array[v + 5] = numpy.array([
            [prev_quads[1].x, prev_quads[1].y, wall_width],
            [curr_quads[1].x, curr_quads[1].y, wall_width + channel_depth],
            [curr_quads[1].x, curr_quads[1].y, wall_width],
        ])

        output_array[v + 6] = numpy.array([
            [prev_quads[1].x, prev_quads[1].y, wall_width],
            [curr_quads[1].x, curr_quads[1].y, wall_width],
            [prev_quads[2].x, prev_quads[2].y, wall_width],
        ])

        output_array[v + 7] = numpy.array([
            [prev_quads[2].x, prev_quads[2].y, wall_width],
            [curr_quads[1].x, curr_quads[1].y, wall_width],
            [curr_quads[2].x, curr_quads[2].y, wall_width],
        ])

        output_array[v + 8] = numpy.array([
            [prev_quads[2].x, prev_quads[2].y, wall_width],
            [curr_quads[2].x, curr_quads[2].y, wall_width],
            [curr_quads[2].x, curr_quads[2].y, wall_width + channel_depth],
        ])

        output_array[v + 9] = numpy.array([
            [prev_quads[2].x, prev_quads[2].y, wall_width],
            [curr_quads[2].x, curr_quads[2].y, wall_width + channel_depth],
            [prev_quads[2].x, prev_quads[2].y, wall_width + channel_depth],
        ])

        output_array[v + 10] = numpy.array([
            [prev_quads[2].x, prev_quads[2].y, wall_width + channel_depth],
            [curr_quads[2].x, curr_quads[2].y, wall_width + channel_depth],
            [prev_quads[3].x, prev_quads[3].y, wall_width + channel_depth],
        ])

        output_array[v + 11] = numpy.array([
            [prev_quads[3].x, prev_quads[3].y, wall_width + channel_depth],
            [curr_quads[2].x, curr_quads[2].y, wall_width + channel_depth],
            [curr_quads[3].x, curr_quads[3].y, wall_width + channel_depth],
        ])

        output_array[v + 12] = numpy.array([
            [prev_quads[3].x, prev_quads[3].y, wall_width + channel_depth],
            [curr_quads[3].x, curr_quads[3].y, wall_width + channel_depth],
            [prev_quads[3].x, prev_quads[3].y, 0],
        ])

        output_array[v + 13] = numpy.array([
            [prev_quads[3].x, prev_quads[3].y, 0],
            [curr_quads[3].x, curr_quads[3].y, wall_width + channel_depth],
            [curr_quads[3].x, curr_quads[3].y, 0],
        ])

        output_array[v + 14] = numpy.array([
            [prev_quads[3].x, prev_quads[3].y, 0],
            [curr_quads[3].x, curr_quads[3].y, 0],
            [prev_quads[0].x, prev_quads[0].y, 0],
        ])

        output_array[v + 15] = numpy.array([
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
    output_array[v] = numpy.array([
        [quads[i][3].x, quads[i][3].y, wall_width + channel_depth],
        [quads[i][2].x, quads[i][2].y, wall_width],
        [quads[i][3].x, quads[i][3].y, 0],
    ])
    output_array[v + 1] = numpy.array([
        [quads[i][3].x, quads[i][3].y, wall_width + channel_depth],
        [quads[i][2].x, quads[i][2].y, wall_width + channel_depth],
        [quads[i][2].x, quads[i][2].y, wall_width],
    ])
    output_array[v + 2] = numpy.array([
        [quads[i][2].x, quads[i][2].y, wall_width],
        [quads[i][0].x, quads[i][0].y, 0],
        [quads[i][3].x, quads[i][3].y, 0],
    ])
    output_array[v + 3] = numpy.array([
        [quads[i][2].x, quads[i][2].y, wall_width],
        [quads[i][1].x, quads[i][1].y, wall_width],
        [quads[i][0].x, quads[i][0].y, 0],
    ])
    output_array[v + 4] = numpy.array([
        [quads[i][1].x, quads[i][1].y, wall_width + channel_depth],
        [quads[i][0].x, quads[i][0].y, wall_width + channel_depth],
        [quads[i][1].x, quads[i][1].y, wall_width],
    ])
    output_array[v + 5] = numpy.array([
        [quads[i][1].x, quads[i][1].y, wall_width],
        [quads[i][0].x, quads[i][0].y, wall_width + channel_depth],
        [quads[i][0].x, quads[i][0].y, 0],
    ])


if __name__ == "__main__":
    main()
