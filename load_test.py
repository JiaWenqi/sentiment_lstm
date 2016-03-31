import unittest
import numpy as np
import tensorflow as tf

import load


class TestNextMiniBatch(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestNextMiniBatch, self).__init__(*args, **kwargs)
    self.x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    self.labels = np.array([0, 1, 2, 3, 4])

  def test_First(self):
    x, labels = load.NextMiniBatch(self.x, self.labels, 0, 2)
    expected_x = np.array([[1, 2], [3, 4]])
    expected_labels = np.array([0, 1])

    np.testing.assert_array_equal(expected_x, x)
    np.testing.assert_array_equal(expected_labels, labels)

  def test_Second(self):
    x, labels = load.NextMiniBatch(self.x, self.labels, 1, 2)
    expected_x = np.array([[5, 6], [7, 8]])
    expected_labels = np.array([2, 3])

    np.testing.assert_array_equal(expected_x, x)
    np.testing.assert_array_equal(expected_labels, labels)

  def test_None(self):
    x, labels = load.NextMiniBatch(self.x, self.labels, 3, 2)

    self.assertIsNone(x)
    self.assertIsNone(labels)


class TestPrepareData(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestPrepareData, self).__init__(*args, **kwargs)

  def test_IntLabel(self):
    seqs = [[1, 2], [1, 2, 3, 4]]
    labels = [0, 1]
    expected_x = np.array([[0, 0, 1, 2], [1, 2, 3, 4]])
    expected_labels = np.array([0, 1])
    actual_x, actual_labels = load.prepare_data(seqs, labels)

    np.testing.assert_array_equal(expected_x, actual_x)
    np.testing.assert_array_equal(expected_labels, actual_labels)

  def test_IntLabel_Truncate(self):
    seqs = [[1, 2], [1, 2, 3, 4]]
    labels = [0, 1]
    expected_x = np.array([[0, 1, 2]])
    expected_labels = np.array([0,])
    actual_x, actual_labels = load.prepare_data(seqs, labels, maxlen=3)

    np.testing.assert_array_equal(expected_x, actual_x)
    np.testing.assert_array_equal(expected_labels, actual_labels)


if __name__ == '__main__':
  unittest.main()
