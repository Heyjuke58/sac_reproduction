import unittest
import os


class Tester(unittest.TestCase):
    def setUp(self) -> None:
        # clear models and results folders before tests:
        for folder_name in ["test/models", "test/results"]:
            file_names = os.listdir(folder_name)

            for file_name in file_names:
                os.remove(os.path.join(folder_name, file_name))
        return super().setUp()
