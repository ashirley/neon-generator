import unittest
from maths import *

class TestMaths(unittest.TestCase):
    # any_lines_intersect
    def test_any_lines_intersect(self):
        result = any_lines_intersect([[1, 0], [3, 1]], [[[2, 2], [3, 0]], [[3, 2], [2, 0]], [[3, 2], [4, 0]]])
        self.assertEqual([0, 1], result)

    def test_any_lines_intersect_vertical(self):
        result = any_lines_intersect(np.array([[1, 0], [3, 1]]), np.array([[[2, 2], [2, 0]]]))
        self.assertEqual([0], result)

    def test_any_lines_intersect_parallel(self):
        result = any_lines_intersect(np.array([[1, 0], [3, 1]]), np.array([[[2, 1], [4, 2]]]))
        self.assertEqual([], result)

    # find_disjoint_subgraphs
    def test_find_disjoint_subgraphs(self):
        result = find_disjoint_subgraphs({
            1: [2, 3],
            2: [3, 4],
            5: [6],
            7: [6]
        })
        self.assertEqual([{1, 2, 3, 4}, {5, 6, 7}], result)


if __name__ == '__main__':
    unittest.main()
