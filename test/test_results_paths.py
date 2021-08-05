import os
import unittest
from tempfile import TemporaryDirectory

from src.results_paths import ResultsPaths


class TestResultsPaths(unittest.TestCase):
    def test_directory_creation(self):
        with TemporaryDirectory(dir="resources") as tmp_dir:
            ResultsPaths(tmp_dir)
            created_files_or_dirs = len(os.listdir(tmp_dir)) != 0
            self.assertTrue(created_files_or_dirs)


if __name__ == '__main__':
    unittest.main()
