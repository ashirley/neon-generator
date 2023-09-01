import numpy as np


# inspired by https://stackoverflow.com/a/74697697
def any_lines_intersect(target, next_lines):
    """Find the indexes of the lines which intersect with the target"""

    old_np_seterr = np.seterr(divide="ignore", invalid="ignore")
    try:
        # print("in target", target)
        # print("in next_lines", next_lines)

        # rearrange and split out the individual coordinates of the lines to test against
        bx1, by1, bx2, by2 = np.array(next_lines).reshape(-1, 4).transpose()

        # print("bx1", bx1)
        # print("bx2", bx2)
        # print("by1", by1)
        # print("by2", by2)

        # take the single target line and copy it to be the same shape as the lines to test against
        ax1 = np.full((len(next_lines)), target[0][0])
        ax2 = np.full((len(next_lines)), target[1][0])
        ay1 = np.full((len(next_lines)), target[0][1])
        ay2 = np.full((len(next_lines)), target[1][1])

        # print("ax1", ax1)
        # print("ax2", ax2)
        # print("ay1", ay1)
        # print("ay2", ay2)

        denom = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)

        # print("denom", denom)

        ua = ((bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)) / denom
        ub = ((ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)) / denom
        hit = np.stack((0.0 <= ua, ua <= 1.0, 0.0 <= ub, ub <= 1.0)).all(axis=0)

        # print("ua", ua)
        # print("ub", ub)
        # print("hit", hit)

        return [i for i, a in enumerate(hit) if a]
    finally:
        np.seterr(**old_np_seterr)


def intersection_position(a, b):
    ax1 = a[0][0]
    ax2 = a[1][0]
    ay1 = a[0][1]
    ay2 = a[1][1]
    bx1 = b[0][0]
    bx2 = b[1][0]
    by1 = b[0][1]
    by2 = b[1][1]

    denom = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)
    ua = ((bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)) / denom
    return ua


def close(a, b):
    """Are the 2 points close to each other?"""
    distance = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    if distance < .1:
        return distance
    else:
        return None


def find_disjoint_subgraphs(graph):
    """Given a dict of links in a graph return a list of sets of disjoint nodes"""
    working_graph = graph.copy()

    # Make sure all links are bi-directional
    for k, v in graph.items():
        for x in v:
            # add the return link from x to k for all links k to x.
            working_graph[x] = set(working_graph.get(x, set())) | {k}

    # print (working_graph)
    disjoint_sets = []
    while len(working_graph.keys()) > 0:
        to_search = set()
        current_disjoint_set = set()
        curr_key = next(iter(working_graph.keys()))

        while True:
            # print ("processing %d" % curr_key)
            current_disjoint_set |= {curr_key}
            to_search |= set(working_graph.pop(curr_key, []))

            if len(to_search) == 0:
                break
            curr_key = to_search.pop()

        # print("Found set %s" % (current_disjoint_set,))
        disjoint_sets.append(current_disjoint_set)
    return disjoint_sets
