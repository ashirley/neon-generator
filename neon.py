from svgelements_monkeypatch import *
from maths import *
import svgwrite
import numpy
from stl import Mesh
import argparse
from collections import namedtuple

debug = False

log_paths = False
log_perp_points = False

debug_perpendicular_style = {"stroke": "green", "stroke_width": "1px"}
debug_original_style = {"stroke": "black", "stroke_width": "3px", "fill": "none"}
debug_comb_style = {"stroke": "blue", "stroke_width": "1px"}
intersection_detection_overlap = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="The SVG file to read")
    parser.add_argument("-l", "--length", help="Target length, the output is scaled so the path is this long. e.g. '260mm' or '500cm'. If not specified, the output won't be scaled")
    parser.add_argument("-r", "--resolution", help="Roughly how many arcs in the whole path, the higher the smoother but bigger output", default=500, type=int)
    parser.add_argument("-o", "--output-file", help="The file to write to", default="output.stl")
    parser.add_argument("-D", "--debug-output-file", help="The file to write a debug SVG to, if this is not supplied debug won't be written")
    parser.add_argument("-d", "--debug", help="Log some additional debug info", action="store_true")
    parser.add_argument("-p", "--profile-size", help="The size of the profile the LED sits in. Either 'noodle', '10mm' or a colon-seperated triplet of <channel_width>:<channel_depth>:<wall_width> in mm e.g. '10:15:2'", type=parse_profile_spec)
    parser.add_argument("-c", "--covered-colour", help="The colour of the line which represents a covered section of the supports e.g. '#ff1414'. If not specified, the input SVG must be a single colour.")

    args = parser.parse_args()

    global debug
    debug = debug or args.debug

    paths = parse_and_validate_stl(args.input_file, args.covered_colour)
    scale = 1

    if args.length:
        total_length = sum(path.length() for path in paths)
        scale = Length(args.length).value(ppi=DEFAULT_PPI) / total_length

        print("scaling '%f' by '%s' to match target length of '%s'" % (total_length, scale, args.length))

    scaled_paths = [Path((path * ("scale(%f)" % scale)).segments(), stroke=path.stroke) for path in paths]
    mm_paths = [Path((path * ("scale(%f)" % (Length("1").to_mm()))).segments(), stroke=path.stroke) for path in scaled_paths]

    if debug:
        print("output path will have a length of %f mm" % sum(path.length() for path in mm_paths))

    # TODO: check for sharp angles
    # TODO: check for overlapping / fix almost joining paths
    # TODO: adjust more than just the last line to meet the next segment.
    #  Especially if it is a short segment, resolution is high or there is a large difference in angle needed.

    perp_points_lists = generate_perpendicular_points(mm_paths, args.resolution, args.profile_size, args.covered_colour)

    (perp_points_lists, error_indexes) = optimise_perp_points_list(perp_points_lists, args.profile_size)
    # print(error_indexes)

    if args.output_file:
        stl_data = numpy.zeros(0, dtype=Mesh.dtype)

        i = 0
        for perp_points_list in perp_points_lists:
            triangles_for_perp_points = create_stl_triangles(perp_points_list[1], args.profile_size)
            stl_data = numpy.concatenate(
                (stl_data, triangles_for_perp_points)
            )

            if perp_points_list[0]:
                stl_data = numpy.concatenate(
                    (stl_data, create_stl_cover_triangles(perp_points_list[1], args.profile_size))
                )
                if debug:
                    print("making cover for perp_points_list index %d" % i)
            i = i+1

        print("Saving %d triangles in %d sections..." % (len(stl_data), len(perp_points_lists)))
        stl_mesh = Mesh(stl_data)
        stl_mesh.save(args.output_file)

    if args.debug_output_file:
        dwg = svgwrite.Drawing(args.debug_output_file, profile='tiny')

        # copy across the original lines
        copy_paths_to_debug_svg(mm_paths, dwg)

        # add the perpendicular lines
        prev_perp_points = None
        g = dwg.g(**debug_perpendicular_style)
        i = 0
        for j, perp_points_list in enumerate(perp_points_lists):
            for perp_points in perp_points_list[1]:

                line = dwg.line(start=(perp_points.lo[0], perp_points.lo[1]), end=(perp_points.ro[0], perp_points.ro[1]))
                if i in error_indexes:
                    line["stroke"] = "red"
                line.set_desc(title="%d: %d"%(j, i))
                g.add(line)

                if prev_perp_points:
                    g.add(dwg.line(start=(prev_perp_points.lo[0], prev_perp_points.lo[1]), end=(perp_points.lo[0], perp_points.lo[1])))
                    g.add(dwg.line(start=(prev_perp_points.ro[0], prev_perp_points.ro[1]), end=(perp_points.ro[0], perp_points.ro[1])))
                i = i+1
                prev_perp_points = perp_points
            prev_perp_points = None
        dwg.add(g)

        # add a curvature comb
        add_curvature_comb_to_debug_svg(mm_paths, dwg, args.resolution)

        dwg.save()

    if debug and log_perp_points:
        for perp_points_list in perp_points_lists:
            for perp_points in perp_points_list:
                print(perp_points)
            print()
            print()


def steps_for_segment(curr_segment, scaled_total_length, resolution):
    return max(2, ceil((curr_segment.length() / scaled_total_length) * resolution))


def parse_and_validate_stl(file, covered_colour):
    svg = SVG.parse(file)
    paths = list(svg.select(lambda e: isinstance(e, Path)))
    stroke_colors = {path.stroke.hex for path in paths}
    if len(paths) == 0:
        print("ERROR: no paths in input SVG")
        raise Exception
    if covered_colour is None:
        if len(stroke_colors) != 1:
            raise Exception("ERROR: multiple color paths in input SVG and no cover color specified")
    else:
        if len(stroke_colors) > 2:
            raise Exception("ERROR: more than 2 color paths in input SVG: %s" % stroke_colors)
        if covered_colour not in stroke_colors:
            raise Exception("ERROR: covered color specified (%s) not present in the input SVG which has (%s)" % (covered_colour, stroke_colors))

    return paths


PerpendicularPoints = namedtuple('PerpendicularPoints', 'lo li lid c rid ri ro')


def generate_perpendicular_points(paths, resolution, profile, covered_colour) -> list[tuple[bool, list[PerpendicularPoints]]]:
    """
    Walk the path, generating a list of 2d coordinates on a line perpendicular to the line of the path.
    The points are for:
     * outer left wall
     * inner left wall
     * inner left wall + delta
     * inner right wall - delta
     * inner right wall
     * outer right wall
    When the elements of the path meet, the angle of the "perpendicular" line is altered to ease the transition
    and allow a neat join

    We actually return a list of lists as we split the paths into contiguous sections. For example, if path contains
    2 non-contiguous straight lines, we will return a list of 2 lists, each inner list will have 2 PerpendicularPoints
    (one for the start of the line and one for the end) with 6 points each.
    """
    scaled_total_length = sum(path.length() for path in paths)

    # TODO: This logic assumes every path starts with a move which doesn't seem right
    perp_points_lists = []
    perp_points_list_num = -1
    for path in paths:
        covered = path.stroke.hex == covered_colour
        prev_join = None  # angle between previous segment and curr
        for i in range(len(path)):
            curr_segment = path[i]  # current segment
            next_segment = path[i + 1] if i + 1 < len(path) else None  # next segment, if any
            curr_join = None  # angle between the current segment and the next

            if isinstance(curr_segment, Move):
                # A move doesn't need neon or a support
                if debug and log_paths:
                    print("Move: %s" % curr_segment)
                if perp_points_list_num == -1 or len(perp_points_lists[perp_points_list_num]) > 0:
                    # There is content in the current list, start a new one.
                    perp_points_list_num += 1
                    perp_points_lists.append((covered, []))
            elif isinstance(curr_segment, Close):
                # TODO
                if debug and log_paths:
                    print("Close: %s" % curr_segment)
            elif isinstance(curr_segment, Line):
                if debug and log_paths:
                    print("Line: %s" % curr_segment)
                # Straight lines only need 2 perpendicular_points along the normal at the start and end
                # (although we only put the end in if the next segment won't duplicate it).
                curr_join = join_angle(next_segment, curr_segment.normal())

                if curr_join is None:
                    # not joining the next segment, add both a start and an end.
                    perp_points_lists[perp_points_list_num][1].extend([
                        perpendicular_points(curr_segment.start, prev_join, curr_segment.normal(), profile),
                        perpendicular_points(curr_segment.end, curr_join, curr_segment.normal(), profile)
                    ])
                else:
                    # joining the next segment, just add a start and leave the next segment to do our end (it's start)
                    perp_points_lists[perp_points_list_num][1].append(perpendicular_points(curr_segment.start, prev_join, curr_segment.normal(), profile))

            elif isinstance(curr_segment, Arc):
                if debug and log_paths:
                    print("Arc: %s" % curr_segment)
                curr_join = join_angle(next_segment, curr_segment.normal(0.99))

                perp_points_lists[perp_points_list_num][1].append(perpendicular_points(curr_segment.point(0), prev_join, curr_segment.normal(0), profile))
                steps = steps_for_segment(curr_segment, scaled_total_length, resolution)
                for j in range(1, steps):
                    perp_points_lists[perp_points_list_num][1].append(perpendicular_points(curr_segment.point(j / steps), None, curr_segment.normal(j / steps), profile))

                if curr_join is None:
                    # not joining the next segment, add an end along the normal of the curve
                    perp_points_lists[perp_points_list_num][1].append(perpendicular_points(curr_segment.point(0.99), curr_join, curr_segment.normal(0.99), profile))

            elif isinstance(curr_segment, QuadraticBezier):
                if debug and log_paths:
                    print("QuadraticBezier: %s" % curr_segment)
                curr_join = join_angle(next_segment, curr_segment.normal(0.99))

                perp_points_lists[perp_points_list_num][1].append(perpendicular_points(curr_segment.point(0), prev_join, curr_segment.normal(0), profile))
                steps = steps_for_segment(curr_segment, scaled_total_length, resolution)
                for j in range(1, steps):
                    perp_points_lists[perp_points_list_num][1].append(perpendicular_points(curr_segment.point(j / steps), None, curr_segment.normal(j / steps), profile))

                if curr_join is None:
                    # not joining the next segment, add an end along the normal of the curve
                    perp_points_lists[perp_points_list_num][1].append(perpendicular_points(curr_segment.point(0.99), curr_join, curr_segment.normal(0.99), profile))

            elif isinstance(curr_segment, CubicBezier):
                if debug and log_paths:
                    print("CubicBezier: %s" % curr_segment)
                curr_join = join_angle(next_segment, curr_segment.normal(0.99))

                perp_points_lists[perp_points_list_num][1].append(perpendicular_points(curr_segment.point(0), prev_join, curr_segment.normal(0), profile))
                steps = steps_for_segment(curr_segment, scaled_total_length, resolution)
                for j in range(1, steps):
                    perp_points_lists[perp_points_list_num][1].append(perpendicular_points(curr_segment.point(j / steps), None, curr_segment.normal(j / steps), profile))

                if curr_join is None:
                    # not joining the next segment, add an end along the normal of the curve
                    perp_points_lists[perp_points_list_num][1].append(perpendicular_points(curr_segment.point(0.99), curr_join, curr_segment.normal(0.99), profile))

            prev_join = curr_join
    return perp_points_lists


def optimise_perp_points_list(perp_points_lists, profile) -> tuple[list[tuple[bool, list[PerpendicularPoints]]], list[int]]:
    """
    Take a list of raw perp_points and optimise for not overlapping each other.
    """

    do_sort = True
    do_deintersect = True

    if do_sort:
        # Before we do anything else, lets arrange paths, so they join up and flatten them.
        # TODO: Do we need to support reversing a path?
        ends = []
        matches = {}
        for perp_points_list in perp_points_lists:
            ends.append(perp_points_list[1][-1].c)

        for i, perp_points_list in enumerate(perp_points_lists):
            start = perp_points_list[1][0].c
            # look for an end which is close enough
            closest_distance = None
            for j, end in enumerate(ends):
                new_distance = close(start, end)
                if new_distance and (not closest_distance or new_distance < closest_distance):
                    closest_distance = new_distance
                    matches[j] = i  # the start should follow this end

        # print(matches)
        # we now have a set of ends which match starts, lets look for a place to begin
        sorted_perp_points_lists = []
        remaining_indexes = set(range(0, len(perp_points_lists)))
        added_so_far = 0
        while len(remaining_indexes) > 0:
            unconnected_starts = remaining_indexes - set(matches.values())
            if len(unconnected_starts) > 0:
                begin = min(unconnected_starts)
            else:
                begin = next(iter(unconnected_starts))

            while True:
                # print(begin, added_so_far, added_so_far + len(perp_points_lists[begin][1]))
                added_so_far += len(perp_points_lists[begin][1])
                sorted_perp_points_lists.append(perp_points_lists[begin])
                remaining_indexes.remove(begin)
                if begin in matches:
                    begin = matches[begin]
                else:
                    break
    else:
        sorted_perp_points_lists = perp_points_lists

    flat_points_list = []
    for perp_points_list in sorted_perp_points_lists:
        flat_points_list += perp_points_list[1]

    # First pass, identify sequences of perp lines which intersect with those nearby.
    error_indexes = set()
    intersect_graph = dict()  # a dict of line indexes which intersect with each other.

    num_points = len(flat_points_list)
    # print(num_points)
    for j in range(0, num_points):
        intersected = any_lines_intersect(
            [flat_points_list[j].lo, flat_points_list[j].ro],
            [[x.lo, x.ro] for x in flat_points_list[min(j + 1, num_points):min(j + intersection_detection_overlap, num_points)]]
        )
        if len(intersected) > 0:
            # print(i, intersected)
            error_indexes.add(j)
            error_indexes |= set([j + x + 1 for x in intersected])

            intersect_graph[j] = [j + x + 1 for x in intersected]

    # Second pass, find disjoint sub graphs of intersecting lines
    disjoint_intersecting_line_sets = find_disjoint_subgraphs(intersect_graph)
    # print(disjoint_intersecting_line_sets)

    # Third pass, for each subgraph, work from the middle out, moving the perp lines so the ends which cross are touching, not crossing

    # TODO: this assumes the values in the set are continuous and all overlap the same way!
    for disjoint_set in disjoint_intersecting_line_sets:
        start = min(disjoint_set)
        midpoint = start + len(disjoint_set) // 2
        # which side of the midpoint do we fix to?
        midpoint_perp_points = flat_points_list[midpoint]
        midpoint_next_perp_points = flat_points_list[midpoint + 1]
        fix_lo = intersection_position([midpoint_perp_points.lo, midpoint_perp_points.ro], [midpoint_next_perp_points.lo, midpoint_next_perp_points.ro]) < 0.5

        # iterate backwards
        for j in range(midpoint - 1, start - 1, -1):
            # move one end of the line at index i on top of the midpoint line instead of intersecting
            old_perp_points = flat_points_list[j]
            if fix_lo:
                new_perp_points = perpendicular_points_from_pair(old_perp_points.ro, midpoint_perp_points.lo, profile)
            else:
                new_perp_points = perpendicular_points_from_pair(midpoint_perp_points.ro, old_perp_points.lo, profile)
            if do_deintersect:
                remaining_index = j
                lists_index = 0
                while len(sorted_perp_points_lists[lists_index][1]) <= remaining_index:
                    remaining_index -= len(sorted_perp_points_lists[lists_index][1])
                    lists_index += 1
                sorted_perp_points_lists[lists_index][1][remaining_index] = new_perp_points


        # iterate forwards
        for j in range(midpoint + 1, start + len(disjoint_set)):

            # move one end of the line at index i on top of the midpoint line instead of intersecting
            old_perp_points = flat_points_list[j]
            if fix_lo:
                new_perp_points = perpendicular_points_from_pair(old_perp_points.ro, midpoint_perp_points.lo, profile)
            else:
                new_perp_points = perpendicular_points_from_pair(midpoint_perp_points.ro, old_perp_points.lo, profile)
            if do_deintersect:
                remaining_index = j
                lists_index = 0
                while len(sorted_perp_points_lists[lists_index][1]) <= remaining_index:
                    remaining_index -= len(sorted_perp_points_lists[lists_index][1])
                    lists_index += 1
                sorted_perp_points_lists[lists_index][1][remaining_index] = new_perp_points

    return sorted_perp_points_lists, sorted(error_indexes)


def copy_paths_to_debug_svg(paths, dwg):
    """
    Copy the paths (as we understand them) out of the source SVG into the debug svg
    """
    min_x = float('inf')
    min_y = float('inf')
    max_x = 0
    max_y = 0

    for path in paths:
        debug_path = svgwrite.path.Path(**debug_original_style)
        for i in range(len(path)):
            curr_segment = path[i]  # current segment
            debug_path.push(curr_segment.d())

        bbox = path.bbox()
        min_x = min(min_x, bbox[0])
        min_y = min(min_y, bbox[1])
        max_x = max(max_x, bbox[2])
        max_y = max(max_y, bbox[3])

        dwg.add(debug_path)

    # add a 5% padding
    x_padding = (max_x - min_x) * 0.05
    y_padding = (max_y - min_y) * 0.05
    min_x = min_x - x_padding
    max_x = max_x + x_padding
    min_y = min_y - y_padding
    max_y = max_y + y_padding

    dwg['viewBox'] = "%d %d %d %d" % (min_x, min_y, max_x - min_x, max_y - min_y)


def add_curvature_comb_to_debug_svg(paths, dwg, resolution):
    """
draw a curvature comb and add it to the debug svg
    """
    scaled_total_length = sum(path.length() for path in paths)
    g = dwg.g(**debug_comb_style)
    for path in paths:
        for i in range(len(path)):
            curr_segment = path[i]  # current segment

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
                steps = steps_for_segment(curr_segment, scaled_total_length, resolution)
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
                steps = steps_for_segment(curr_segment, scaled_total_length, resolution)
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
                steps = steps_for_segment(curr_segment, scaled_total_length, resolution)
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


ProfileSpec = namedtuple('ProfileSpec', 'channel_width channel_depth wall_width')


def parse_profile_spec(spec):
    if spec == "noodle":
        return ProfileSpec(2, 2, 1)
    elif spec == "10mm":
        return ProfileSpec(10, 15, 2)
    else:
        # is it 3, colon separated, numbers
        m = re.search("^([0-9.]+):([0-9.]+):([0-9.]+)$", spec)
        if m is None:
            raise argparse.ArgumentTypeError("Profile spec should be 3 numbers separated by colons, units are not allowed")
        return ProfileSpec(float(m.group(1)), float(m.group(2)), float(m.group(3)))


# generate points along the join line, centered on the given point with the given normal.
def perpendicular_points(center, join, normal, profile) -> PerpendicularPoints:
    # find the angle between the join line and the normal
    # 1/cos(angle) is how much bigger the distances should be between center and the perpendicular points
    if join is None:
        join = normal

    angle = Point(0).angle_to(normal).as_radians - Point(0).angle_to(join).as_radians
    multiple = 1 / cos(angle)

    a = join * multiple * (profile.channel_width / 2)
    b = join * multiple * (profile.channel_width / 2 + profile.wall_width)
    delta = join * multiple * 0.2
    return PerpendicularPoints(center + b, center + a, center + a - delta, center, center - a + delta, center - a, center - b)


# generate points along the join line, centered on the given point with the given normal.
def perpendicular_points_from_pair(m, n, profile) -> PerpendicularPoints:
    # Under normal circumstances, the line is of length (profile.channel_width + 2 * profile.wall_width), find the multiple we're aiming for
    typical_length = (profile.channel_width + 2 * profile.wall_width)
    actual_length = abs(n - m)
    multiple = actual_length / typical_length
    # TODO: This feels like it is squishing the channel in an unhelpful way?
    # print("multiple", multiple)

    angle = (n - m)
    angle = angle * (1 / abs(angle))  # make a unit vector
    center = 0.5 * (n + m)

    a = angle * multiple * (profile.channel_width / 2)
    b = angle * multiple * (profile.channel_width / 2 + profile.wall_width)
    delta = angle * multiple * 0.2
    return PerpendicularPoints(center + b, center + a, center + a - delta, center, center - a + delta, center - a, center - b)


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
    else:
        return None

    # make a unit vector
    return curr_join * (1 / abs(curr_join))


def create_stl_triangles(perp_points_list: list[PerpendicularPoints], profile):
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

    # Each perp_points is 8 vertices (2 z positions for each 4 2d points)
    # between each set of 8 vertices is 8 square faces = 16 triangles)
    # At the end is 6 triangles for the 1 face.
    num_triangles_in_ppl = (len(perp_points_list) - 1) * 16 + 12
    output = numpy.zeros(num_triangles_in_ppl, dtype=Mesh.dtype)
    output_array = output['vectors']

    output_array[0] = numpy.array([
        [perp_points_list[0].lo.x, perp_points_list[0].lo.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[0].li.x, perp_points_list[0].li.y, profile.wall_width],
        [perp_points_list[0].lo.x, perp_points_list[0].lo.y, 0],
    ])
    output_array[1] = numpy.array([
        [perp_points_list[0].lo.x, perp_points_list[0].lo.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[0].li.x, perp_points_list[0].li.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[0].li.x, perp_points_list[0].li.y, profile.wall_width],
    ])
    output_array[2] = numpy.array([
        [perp_points_list[0].li.x, perp_points_list[0].li.y, profile.wall_width],
        [perp_points_list[0].ro.x, perp_points_list[0].ro.y, 0],
        [perp_points_list[0].lo.x, perp_points_list[0].lo.y, 0],
    ])
    output_array[3] = numpy.array([
        [perp_points_list[0].li.x, perp_points_list[0].li.y, profile.wall_width],
        [perp_points_list[0].ri.x, perp_points_list[0].ri.y, profile.wall_width],
        [perp_points_list[0].ro.x, perp_points_list[0].ro.y, 0],
    ])
    output_array[4] = numpy.array([
        [perp_points_list[0].ri.x, perp_points_list[0].ri.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[0].ro.x, perp_points_list[0].ro.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[0].ri.x, perp_points_list[0].ri.y, profile.wall_width],
    ])
    output_array[5] = numpy.array([
        [perp_points_list[0].ri.x, perp_points_list[0].ri.y, profile.wall_width],
        [perp_points_list[0].ro.x, perp_points_list[0].ro.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[0].ro.x, perp_points_list[0].ro.y, 0],
    ])
    for i in range(1, len(perp_points_list)):
        prev_perp_points = perp_points_list[i - 1]
        curr_perp_points = perp_points_list[i]

        v = (i - 1) * 16 + 6

        output_array[v] = numpy.array([
            [prev_perp_points.lo.x, prev_perp_points.lo.y, 0],
            [curr_perp_points.lo.x, curr_perp_points.lo.y, 0],
            [prev_perp_points.lo.x, prev_perp_points.lo.y, profile.wall_width + profile.channel_depth],
        ])

        output_array[v + 1] = numpy.array([
            [prev_perp_points.lo.x, prev_perp_points.lo.y, profile.wall_width + profile.channel_depth],
            [curr_perp_points.lo.x, curr_perp_points.lo.y, 0],
            [curr_perp_points.lo.x, curr_perp_points.lo.y, profile.wall_width + profile.channel_depth],
        ])

        output_array[v + 2] = numpy.array([
            [prev_perp_points.lo.x, prev_perp_points.lo.y, profile.wall_width + profile.channel_depth],
            [curr_perp_points.lo.x, curr_perp_points.lo.y, profile.wall_width + profile.channel_depth],
            [curr_perp_points.li.x, curr_perp_points.li.y, profile.wall_width + profile.channel_depth],
        ])

        output_array[v + 3] = numpy.array([
            [prev_perp_points.lo.x, prev_perp_points.lo.y, profile.wall_width + profile.channel_depth],
            [curr_perp_points.li.x, curr_perp_points.li.y, profile.wall_width + profile.channel_depth],
            [prev_perp_points.li.x, prev_perp_points.li.y, profile.wall_width + profile.channel_depth],
        ])

        output_array[v + 4] = numpy.array([
            [prev_perp_points.li.x, prev_perp_points.li.y, profile.wall_width + profile.channel_depth],
            [curr_perp_points.li.x, curr_perp_points.li.y, profile.wall_width + profile.channel_depth],
            [prev_perp_points.li.x, prev_perp_points.li.y, profile.wall_width],
        ])

        output_array[v + 5] = numpy.array([
            [prev_perp_points.li.x, prev_perp_points.li.y, profile.wall_width],
            [curr_perp_points.li.x, curr_perp_points.li.y, profile.wall_width + profile.channel_depth],
            [curr_perp_points.li.x, curr_perp_points.li.y, profile.wall_width],
        ])

        output_array[v + 6] = numpy.array([
            [prev_perp_points.li.x, prev_perp_points.li.y, profile.wall_width],
            [curr_perp_points.li.x, curr_perp_points.li.y, profile.wall_width],
            [prev_perp_points.ri.x, prev_perp_points.ri.y, profile.wall_width],
        ])

        output_array[v + 7] = numpy.array([
            [prev_perp_points.ri.x, prev_perp_points.ri.y, profile.wall_width],
            [curr_perp_points.li.x, curr_perp_points.li.y, profile.wall_width],
            [curr_perp_points.ri.x, curr_perp_points.ri.y, profile.wall_width],
        ])

        output_array[v + 8] = numpy.array([
            [prev_perp_points.ri.x, prev_perp_points.ri.y, profile.wall_width],
            [curr_perp_points.ri.x, curr_perp_points.ri.y, profile.wall_width],
            [curr_perp_points.ri.x, curr_perp_points.ri.y, profile.wall_width + profile.channel_depth],
        ])

        output_array[v + 9] = numpy.array([
            [prev_perp_points.ri.x, prev_perp_points.ri.y, profile.wall_width],
            [curr_perp_points.ri.x, curr_perp_points.ri.y, profile.wall_width + profile.channel_depth],
            [prev_perp_points.ri.x, prev_perp_points.ri.y, profile.wall_width + profile.channel_depth],
        ])

        output_array[v + 10] = numpy.array([
            [prev_perp_points.ri.x, prev_perp_points.ri.y, profile.wall_width + profile.channel_depth],
            [curr_perp_points.ri.x, curr_perp_points.ri.y, profile.wall_width + profile.channel_depth],
            [prev_perp_points.ro.x, prev_perp_points.ro.y, profile.wall_width + profile.channel_depth],
        ])

        output_array[v + 11] = numpy.array([
            [prev_perp_points.ro.x, prev_perp_points.ro.y, profile.wall_width + profile.channel_depth],
            [curr_perp_points.ri.x, curr_perp_points.ri.y, profile.wall_width + profile.channel_depth],
            [curr_perp_points.ro.x, curr_perp_points.ro.y, profile.wall_width + profile.channel_depth],
        ])

        output_array[v + 12] = numpy.array([
            [prev_perp_points.ro.x, prev_perp_points.ro.y, profile.wall_width + profile.channel_depth],
            [curr_perp_points.ro.x, curr_perp_points.ro.y, profile.wall_width + profile.channel_depth],
            [prev_perp_points.ro.x, prev_perp_points.ro.y, 0],
        ])

        output_array[v + 13] = numpy.array([
            [prev_perp_points.ro.x, prev_perp_points.ro.y, 0],
            [curr_perp_points.ro.x, curr_perp_points.ro.y, profile.wall_width + profile.channel_depth],
            [curr_perp_points.ro.x, curr_perp_points.ro.y, 0],
        ])

        output_array[v + 14] = numpy.array([
            [prev_perp_points.ro.x, prev_perp_points.ro.y, 0],
            [curr_perp_points.ro.x, curr_perp_points.ro.y, 0],
            [prev_perp_points.lo.x, prev_perp_points.lo.y, 0],
        ])

        output_array[v + 15] = numpy.array([
            [curr_perp_points.ro.x, curr_perp_points.ro.y, 0],
            [curr_perp_points.lo.x, curr_perp_points.lo.y, 0],
            [prev_perp_points.lo.x, prev_perp_points.lo.y, 0],
        ])
    #  ---    ---    wall_width + channel_depth
    # |0\1|__|4/ |   wall_width
    # |  / \  \ 5|
    # | / 2  \3\ |
    # |/_______\\|   0
    #
    # 3   2  1   0
    i = len(perp_points_list) - 1
    v = i * 16 + 6
    output_array[v] = numpy.array([
        [perp_points_list[i].ro.x, perp_points_list[i].ro.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[i].ri.x, perp_points_list[i].ri.y, profile.wall_width],
        [perp_points_list[i].ro.x, perp_points_list[i].ro.y, 0],
    ])
    output_array[v + 1] = numpy.array([
        [perp_points_list[i].ro.x, perp_points_list[i].ro.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[i].ri.x, perp_points_list[i].ri.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[i].ri.x, perp_points_list[i].ri.y, profile.wall_width],
    ])
    output_array[v + 2] = numpy.array([
        [perp_points_list[i].ri.x, perp_points_list[i].ri.y, profile.wall_width],
        [perp_points_list[i].lo.x, perp_points_list[i].lo.y, 0],
        [perp_points_list[i].ro.x, perp_points_list[i].ro.y, 0],
    ])
    output_array[v + 3] = numpy.array([
        [perp_points_list[i].ri.x, perp_points_list[i].ri.y, profile.wall_width],
        [perp_points_list[i].li.x, perp_points_list[i].li.y, profile.wall_width],
        [perp_points_list[i].lo.x, perp_points_list[i].lo.y, 0],
    ])
    output_array[v + 4] = numpy.array([
        [perp_points_list[i].li.x, perp_points_list[i].li.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[i].lo.x, perp_points_list[i].lo.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[i].li.x, perp_points_list[i].li.y, profile.wall_width],
    ])
    output_array[v + 5] = numpy.array([
        [perp_points_list[i].li.x, perp_points_list[i].li.y, profile.wall_width],
        [perp_points_list[i].lo.x, perp_points_list[i].lo.y, profile.wall_width + profile.channel_depth],
        [perp_points_list[i].lo.x, perp_points_list[i].lo.y, 0],
    ])

    return output


def create_stl_cover_triangles(perp_points_list, profile):
    """
    Create the triangles for the stl for a cover for a single contiguous path.

    """
    #  ------------      0
    # |\ 1/|\4|\6 /|
    # |0\/2|3\|5\/7|
    #  -- 8---- 9--      wall_width
    #    |/    \|        wall_width + a
    #
    # |_ 0
    # |__ wall_width + 0.2
    # |____ wall_width + 0.2 + b

    # Each perp_points is 10 vertices
    # between each set of 10 vertices is 12 square faces = 24 triangles)
    # At the end is 10 triangles for the 1 face.
    num_triangles_in_ppl = (len(perp_points_list) - 1) * 24 + 20
    output = numpy.zeros(num_triangles_in_ppl, dtype=Mesh.dtype)
    output_array = output['vectors']

    a = 0.2

    offset_x = 50
    offset_y = 0

    output_array[0] = numpy.array([
        [perp_points_list[0].lo.x + offset_x, perp_points_list[0].lo.y + offset_y, 0],
        [perp_points_list[0].lo.x + offset_x, perp_points_list[0].lo.y + offset_y, profile.wall_width],
        [perp_points_list[0].li.x + offset_x, perp_points_list[0].li.y + offset_y, profile.wall_width],
    ])

    output_array[1] = numpy.array([
        [perp_points_list[0].lo.x + offset_x, perp_points_list[0].lo.y + offset_y, 0],
        [perp_points_list[0].li.x + offset_x, perp_points_list[0].li.y + offset_y, profile.wall_width],
        [perp_points_list[0].lid.x + offset_x, perp_points_list[0].lid.y + offset_y, 0],
    ])

    output_array[2] = numpy.array([
        [perp_points_list[0].lid.x + offset_x, perp_points_list[0].lid.y + offset_y, 0],
        [perp_points_list[0].li.x + offset_x, perp_points_list[0].li.y + offset_y, profile.wall_width],
        [perp_points_list[0].lid.x + offset_x, perp_points_list[0].lid.y + offset_y, profile.wall_width],
    ])

    output_array[3] = numpy.array([
        [perp_points_list[0].lid.x + offset_x, perp_points_list[0].lid.y + offset_y, 0],
        [perp_points_list[0].lid.x + offset_x, perp_points_list[0].lid.y + offset_y, profile.wall_width],
        [perp_points_list[0].rid.x + offset_x, perp_points_list[0].rid.y + offset_y, profile.wall_width],
    ])

    output_array[4] = numpy.array([
        [perp_points_list[0].rid.x + offset_x, perp_points_list[0].rid.y + offset_y, 0],
        [perp_points_list[0].lid.x + offset_x, perp_points_list[0].lid.y + offset_y, 0],
        [perp_points_list[0].rid.x + offset_x, perp_points_list[0].rid.y + offset_y, profile.wall_width],
    ])

    output_array[5] = numpy.array([
        [perp_points_list[0].rid.x + offset_x, perp_points_list[0].rid.y + offset_y, 0],
        [perp_points_list[0].rid.x + offset_x, perp_points_list[0].rid.y + offset_y, profile.wall_width],
        [perp_points_list[0].ri.x + offset_x, perp_points_list[0].ri.y + offset_y, profile.wall_width],
    ])

    output_array[6] = numpy.array([
        [perp_points_list[0].rid.x + offset_x, perp_points_list[0].rid.y + offset_y, 0],
        [perp_points_list[0].ri.x + offset_x, perp_points_list[0].ri.y + offset_y, profile.wall_width],
        [perp_points_list[0].ro.x + offset_x, perp_points_list[0].ro.y + offset_y, 0],
    ])

    output_array[7] = numpy.array([
        [perp_points_list[0].ro.x + offset_x, perp_points_list[0].ro.y + offset_y, 0],
        [perp_points_list[0].ri.x + offset_x, perp_points_list[0].ri.y + offset_y, profile.wall_width],
        [perp_points_list[0].ro.x + offset_x, perp_points_list[0].ro.y + offset_y, profile.wall_width],
    ])

    output_array[8] = numpy.array([
        [perp_points_list[0].li.x + offset_x, perp_points_list[0].li.y + offset_y, profile.wall_width],
        [perp_points_list[0].li.x + offset_x, perp_points_list[0].li.y + offset_y, profile.wall_width + a],
        [perp_points_list[0].lid.x + offset_x, perp_points_list[0].lid.y + offset_y, profile.wall_width],
    ])

    output_array[9] = numpy.array([
        [perp_points_list[0].ri.x + offset_x, perp_points_list[0].ri.y + offset_y, profile.wall_width],
        [perp_points_list[0].rid.x + offset_x, perp_points_list[0].rid.y + offset_y, profile.wall_width],
        [perp_points_list[0].ri.x + offset_x, perp_points_list[0].ri.y + offset_y, profile.wall_width + a],
    ])

    for i in range(1, len(perp_points_list)):
        prev_perp_points = perp_points_list[i - 1]
        curr_perp_points = perp_points_list[i]

        v = (i - 1) * 24 + 10

        output_array[v] = numpy.array([
            [prev_perp_points.rid.x + offset_x, prev_perp_points.rid.y + offset_y, 0],
            [prev_perp_points.ro.x + offset_x, prev_perp_points.ro.y + offset_y, 0],
            [curr_perp_points.rid.x + offset_x, curr_perp_points.rid.y + offset_y, 0],
        ])

        output_array[v + 1] = numpy.array([
            [prev_perp_points.ro.x + offset_x, prev_perp_points.ro.y + offset_y, 0],
            [curr_perp_points.ro.x + offset_x, curr_perp_points.ro.y + offset_y, 0],
            [curr_perp_points.rid.x + offset_x, curr_perp_points.rid.y + offset_y, 0],
        ])

        output_array[v + 2] = numpy.array([
            [prev_perp_points.lid.x + offset_x, prev_perp_points.lid.y + offset_y, 0],
            [prev_perp_points.rid.x + offset_x, prev_perp_points.rid.y + offset_y, 0],
            [curr_perp_points.lid.x + offset_x, curr_perp_points.lid.y + offset_y, 0],
        ])

        output_array[v + 3] = numpy.array([
            [prev_perp_points.rid.x + offset_x, prev_perp_points.rid.y + offset_y, 0],
            [curr_perp_points.rid.x + offset_x, curr_perp_points.rid.y + offset_y, 0],
            [curr_perp_points.lid.x + offset_x, curr_perp_points.lid.y + offset_y, 0],
        ])

        output_array[v + 4] = numpy.array([
            [prev_perp_points.lo.x + offset_x, prev_perp_points.lo.y + offset_y, 0],
            [prev_perp_points.lid.x + offset_x, prev_perp_points.lid.y + offset_y, 0],
            [curr_perp_points.lo.x + offset_x, curr_perp_points.lo.y + offset_y, 0],
        ])

        output_array[v + 5] = numpy.array([
            [prev_perp_points.lid.x + offset_x, prev_perp_points.lid.y + offset_y, 0],
            [curr_perp_points.lid.x + offset_x, curr_perp_points.lid.y + offset_y, 0],
            [curr_perp_points.lo.x + offset_x, curr_perp_points.lo.y + offset_y, 0],
        ])

        output_array[v + 6] = numpy.array([
            [prev_perp_points.ro.x + offset_x, prev_perp_points.ro.y + offset_y, 0],
            [prev_perp_points.ro.x + offset_x, prev_perp_points.ro.y + offset_y, profile.wall_width],
            [curr_perp_points.ro.x + offset_x, curr_perp_points.ro.y + offset_y, 0],
        ])

        output_array[v + 7] = numpy.array([
            [prev_perp_points.ro.x + offset_x, prev_perp_points.ro.y + offset_y, profile.wall_width],
            [curr_perp_points.ro.x + offset_x, curr_perp_points.ro.y + offset_y, profile.wall_width],
            [curr_perp_points.ro.x + offset_x, curr_perp_points.ro.y + offset_y, 0],
        ])

        output_array[v + 8] = numpy.array([
            [prev_perp_points.ro.x + offset_x, prev_perp_points.ro.y + offset_y, profile.wall_width],
            [prev_perp_points.ri.x + offset_x, prev_perp_points.ri.y + offset_y, profile.wall_width],
            [curr_perp_points.ro.x + offset_x, curr_perp_points.ro.y + offset_y, profile.wall_width],
        ])

        output_array[v + 9] = numpy.array([
            [curr_perp_points.ri.x + offset_x, curr_perp_points.ri.y + offset_y, profile.wall_width],
            [curr_perp_points.ro.x + offset_x, curr_perp_points.ro.y + offset_y, profile.wall_width],
            [prev_perp_points.ri.x + offset_x, prev_perp_points.ri.y + offset_y, profile.wall_width],
        ])

        output_array[v + 10] = numpy.array([
            [prev_perp_points.ri.x + offset_x, prev_perp_points.ri.y + offset_y, profile.wall_width],
            [prev_perp_points.ri.x + offset_x, prev_perp_points.ri.y + offset_y, profile.wall_width + a],
            [curr_perp_points.ri.x + offset_x, curr_perp_points.ri.y + offset_y, profile.wall_width],
        ])

        output_array[v + 11] = numpy.array([
            [prev_perp_points.ri.x + offset_x, prev_perp_points.ri.y + offset_y, profile.wall_width + a],
            [curr_perp_points.ri.x + offset_x, curr_perp_points.ri.y + offset_y, profile.wall_width + a],
            [curr_perp_points.ri.x + offset_x, curr_perp_points.ri.y + offset_y, profile.wall_width],
        ])

        output_array[v + 12] = numpy.array([
            [prev_perp_points.rid.x + offset_x, prev_perp_points.rid.y + offset_y, profile.wall_width],
            [curr_perp_points.rid.x + offset_x, curr_perp_points.rid.y + offset_y, profile.wall_width],
            [prev_perp_points.ri.x + offset_x, prev_perp_points.ri.y + offset_y, profile.wall_width + a],
        ])

        output_array[v + 13] = numpy.array([
            [prev_perp_points.ri.x + offset_x, prev_perp_points.ri.y + offset_y, profile.wall_width + a],
            [curr_perp_points.rid.x + offset_x, curr_perp_points.rid.y + offset_y, profile.wall_width],
            [curr_perp_points.ri.x + offset_x, curr_perp_points.ri.y + offset_y, profile.wall_width + a],
        ])

        output_array[v + 14] = numpy.array([
            [prev_perp_points.rid.x + offset_x, prev_perp_points.rid.y + offset_y, profile.wall_width],
            [prev_perp_points.lid.x + offset_x, prev_perp_points.lid.y + offset_y, profile.wall_width],
            [curr_perp_points.rid.x + offset_x, curr_perp_points.rid.y + offset_y, profile.wall_width],
        ])

        output_array[v + 15] = numpy.array([
            [curr_perp_points.lid.x + offset_x, curr_perp_points.lid.y + offset_y, profile.wall_width],
            [curr_perp_points.rid.x + offset_x, curr_perp_points.rid.y + offset_y, profile.wall_width],
            [prev_perp_points.lid.x + offset_x, prev_perp_points.lid.y + offset_y, profile.wall_width],
        ])

        output_array[v + 16] = numpy.array([
            [prev_perp_points.lid.x + offset_x, prev_perp_points.lid.y + offset_y, profile.wall_width],
            [prev_perp_points.li.x + offset_x, prev_perp_points.li.y + offset_y, profile.wall_width + a],
            [curr_perp_points.lid.x + offset_x, curr_perp_points.lid.y + offset_y, profile.wall_width],
        ])

        output_array[v + 17] = numpy.array([
            [prev_perp_points.li.x + offset_x, prev_perp_points.li.y + offset_y, profile.wall_width + a],
            [curr_perp_points.li.x + offset_x, curr_perp_points.li.y + offset_y, profile.wall_width + a],
            [curr_perp_points.lid.x + offset_x, curr_perp_points.lid.y + offset_y, profile.wall_width],
        ])

        output_array[v + 18] = numpy.array([
            [prev_perp_points.li.x + offset_x, prev_perp_points.li.y + offset_y, profile.wall_width],
            [curr_perp_points.li.x + offset_x, curr_perp_points.li.y + offset_y, profile.wall_width],
            [prev_perp_points.li.x + offset_x, prev_perp_points.li.y + offset_y, profile.wall_width + a],
        ])

        output_array[v + 19] = numpy.array([
            [prev_perp_points.li.x + offset_x, prev_perp_points.li.y + offset_y, profile.wall_width + a],
            [curr_perp_points.li.x + offset_x, curr_perp_points.li.y + offset_y, profile.wall_width],
            [curr_perp_points.li.x + offset_x, curr_perp_points.li.y + offset_y, profile.wall_width + a],
        ])

        output_array[v + 20] = numpy.array([
            [prev_perp_points.li.x + offset_x, prev_perp_points.li.y + offset_y, profile.wall_width],
            [prev_perp_points.lo.x + offset_x, prev_perp_points.lo.y + offset_y, profile.wall_width],
            [curr_perp_points.li.x + offset_x, curr_perp_points.li.y + offset_y, profile.wall_width],
        ])

        output_array[v + 21] = numpy.array([
            [curr_perp_points.lo.x + offset_x, curr_perp_points.lo.y + offset_y, profile.wall_width],
            [curr_perp_points.li.x + offset_x, curr_perp_points.li.y + offset_y, profile.wall_width],
            [prev_perp_points.lo.x + offset_x, prev_perp_points.lo.y + offset_y, profile.wall_width],
        ])

        output_array[v + 22] = numpy.array([
            [prev_perp_points.lo.x + offset_x, prev_perp_points.lo.y + offset_y, 0],
            [curr_perp_points.lo.x + offset_x, curr_perp_points.lo.y + offset_y, 0],
            [prev_perp_points.lo.x + offset_x, prev_perp_points.lo.y + offset_y, profile.wall_width],
        ])

        output_array[v + 23] = numpy.array([
            [prev_perp_points.lo.x + offset_x, prev_perp_points.lo.y + offset_y, profile.wall_width],
            [curr_perp_points.lo.x + offset_x, curr_perp_points.lo.y + offset_y, 0],
            [curr_perp_points.lo.x + offset_x, curr_perp_points.lo.y + offset_y, profile.wall_width],
        ])

    i = len(perp_points_list) - 1
    v = i * 24 + 10

    output_array[v] = numpy.array([
        [perp_points_list[i].lo.x + offset_x, perp_points_list[i].lo.y + offset_y, 0],
        [perp_points_list[i].li.x + offset_x, perp_points_list[i].li.y + offset_y, profile.wall_width],
        [perp_points_list[i].lo.x + offset_x, perp_points_list[i].lo.y + offset_y, profile.wall_width],
    ])

    output_array[v + 1] = numpy.array([
        [perp_points_list[i].lo.x + offset_x, perp_points_list[i].lo.y + offset_y, 0],
        [perp_points_list[i].lid.x + offset_x, perp_points_list[i].lid.y + offset_y, 0],
        [perp_points_list[i].li.x + offset_x, perp_points_list[i].li.y + offset_y, profile.wall_width],
    ])

    output_array[v + 2] = numpy.array([
        [perp_points_list[i].lid.x + offset_x, perp_points_list[i].lid.y + offset_y, 0],
        [perp_points_list[i].lid.x + offset_x, perp_points_list[i].lid.y + offset_y, profile.wall_width],
        [perp_points_list[i].li.x + offset_x, perp_points_list[i].li.y + offset_y, profile.wall_width],
    ])

    output_array[v + 3] = numpy.array([
        [perp_points_list[i].lid.x + offset_x, perp_points_list[i].lid.y + offset_y, 0],
        [perp_points_list[i].rid.x + offset_x, perp_points_list[i].rid.y + offset_y, profile.wall_width],
        [perp_points_list[i].lid.x + offset_x, perp_points_list[i].lid.y + offset_y, profile.wall_width],
    ])

    output_array[v + 4] = numpy.array([
        [perp_points_list[i].rid.x + offset_x, perp_points_list[i].rid.y + offset_y, 0],
        [perp_points_list[i].rid.x + offset_x, perp_points_list[i].rid.y + offset_y, profile.wall_width],
        [perp_points_list[i].lid.x + offset_x, perp_points_list[i].lid.y + offset_y, 0],
    ])

    output_array[v + 5] = numpy.array([
        [perp_points_list[i].rid.x + offset_x, perp_points_list[i].rid.y + offset_y, 0],
        [perp_points_list[i].ri.x + offset_x, perp_points_list[i].ri.y + offset_y, profile.wall_width],
        [perp_points_list[i].rid.x + offset_x, perp_points_list[i].rid.y + offset_y, profile.wall_width],
    ])

    output_array[v + 6] = numpy.array([
        [perp_points_list[i].rid.x + offset_x, perp_points_list[i].rid.y + offset_y, 0],
        [perp_points_list[i].ro.x + offset_x, perp_points_list[i].ro.y + offset_y, 0],
        [perp_points_list[i].ri.x + offset_x, perp_points_list[i].ri.y + offset_y, profile.wall_width],
    ])

    output_array[v + 7] = numpy.array([
        [perp_points_list[i].ro.x + offset_x, perp_points_list[i].ro.y + offset_y, 0],
        [perp_points_list[i].ro.x + offset_x, perp_points_list[i].ro.y + offset_y, profile.wall_width],
        [perp_points_list[i].ri.x + offset_x, perp_points_list[i].ri.y + offset_y, profile.wall_width],
    ])

    output_array[v + 8] = numpy.array([
        [perp_points_list[i].li.x + offset_x, perp_points_list[i].li.y + offset_y, profile.wall_width],
        [perp_points_list[i].lid.x + offset_x, perp_points_list[i].lid.y + offset_y, profile.wall_width],
        [perp_points_list[i].li.x + offset_x, perp_points_list[i].li.y + offset_y, profile.wall_width + a],
    ])

    output_array[v + 9] = numpy.array([
        [perp_points_list[i].ri.x + offset_x, perp_points_list[i].ri.y + offset_y, profile.wall_width],
        [perp_points_list[i].ri.x + offset_x, perp_points_list[i].ri.y + offset_y, profile.wall_width + a],
        [perp_points_list[i].rid.x + offset_x, perp_points_list[i].rid.y + offset_y, profile.wall_width],
    ])

    return output


if __name__ == "__main__":
    main()
