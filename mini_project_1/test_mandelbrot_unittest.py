import unittest
import numpy as np

from utils.functions import naive, vectorized, numba, vectorized_block


class TestMandelbrotImplementations(unittest.TestCase):
    def setUp(self):
        self.params = (-2.0, 1.0, -1.5, 1.5, 64, 64, 80)

    def test_output_shape_matches_requested_size(self):
        out = vectorized(*self.params)
        self.assertEqual(out.shape, (64, 64))

    def test_vectorized_matches_naive_up_to_escape_index_convention(self):
        naive_out = naive(*self.params)
        vec_out = vectorized(*self.params)
        max_iter = self.params[-1]
        escaped = vec_out < max_iter
        np.testing.assert_array_equal(naive_out[escaped], vec_out[escaped] + 1)
        np.testing.assert_array_equal(naive_out[~escaped], vec_out[~escaped])

    def test_numba_matches_naive(self):
        # first call compiles numba function
        numba(*self.params)
        numba_out = numba(*self.params)
        naive_out = naive(*self.params)
        np.testing.assert_array_equal(numba_out, naive_out)

    def test_vectorized_block_matches_full_vectorized_subset(self):
        xmin, xmax, ymin, ymax, height, width, max_iter = self.params
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)

        block_start, block_end = 10, 20
        y_block = y[block_start:block_end]

        block_out = vectorized_block(y_block, x, max_iter)
        full_out = vectorized(*self.params)

        np.testing.assert_array_equal(block_out, full_out[block_start:block_end, :])


if __name__ == "__main__":
    unittest.main()
