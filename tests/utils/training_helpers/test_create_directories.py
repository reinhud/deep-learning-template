from pathlib import Path
from typing import List

import pytest

from src.utils.training_helpers.create_directories import create_directories


class TestCreateDirectories:
    @pytest.mark.parametrize(
        "paths, expected_created",
        [
            (["path1", "path2", "path3"], True),
            (["path1", None, "path3"], True),
            ([], False),
        ],
    )
    def test_create_directories(self, tmp_path, paths: List[str | None], expected_created: bool):
        """Test create_directories function with different input paths.

        This test checks the behavior of the create_directories function when provided with various
        input paths, including both existing and non-existing paths, and None values.

        :param tmp_path: The pytest tmp_path fixture.
        :param paths: A list of directory paths to be created.
        :param expected_created: A boolean indicating whether the directory is expected to be
            created or not.
        """
        # Create the temporary dir paths
        paths_list = []
        for p in paths:
            if p is not None:
                tmp_dir_path = tmp_path / p
                paths_list.append(tmp_dir_path)
            else:
                paths_list.append(None)

        # Create the directories
        create_directories(paths_list)

        # Check if the directories were created
        for path in paths_list:
            if path is not None:
                assert Path(path).exists() == expected_created
