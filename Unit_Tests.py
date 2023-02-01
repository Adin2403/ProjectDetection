import unittest
import cv2
import numpy as np

from main import detect_object, draw_rectangle


class TestObjectDetection(unittest.TestCase):
    def test_detect_object(self):
        object_image = cv2.imread("shape1.jpg")
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        frame[100:100+object_image.shape[0], 100:100+object_image.shape[1]] = object_image

        top_left, bottom_right = detect_object(frame, object_image, threshold=0.8)

        self.assertIsNotNone(top_left)
        self.assertIsNotNone(bottom_right)
        self.assertTupleEqual(top_left, (100, 100))
        self.assertTupleEqual(bottom_right, (100 + object_image.shape[1], 100 + object_image.shape[0]))

    def test_detect_object_not_found(self):
        object_image = cv2.imread("shape1.jpg")
        frame = np.zeros((600, 800, 3), dtype=np.uint8)

        top_left, bottom_right = detect_object(frame, object_image, threshold=0.8)

        self.assertIsNone(top_left)
        self.assertIsNone(bottom_right)

    def test_draw_rectangle(self):
        frame = np.zeros((600, 800, 3), dtype=np.uint8)

        draw_rectangle(frame, (100, 100), (200, 200))

        self.assertTrue((frame[100, 100] == [0, 0, 255]).all())
        self.assertTrue((frame[100, 199] == [0, 0, 255]).all())
        self.assertTrue((frame[199, 100] == [0, 0, 255]).all())
        self.assertTrue((frame[199, 199] == [0, 0, 255]).all())

if __name__ == '__main__':
    unittest.main()
