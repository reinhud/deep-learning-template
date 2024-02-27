'''from pathlib import Path

import pytest

from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]"]

# TODO: Fix this test
@pytest.mark.slow
def test_experiments(tmp_path: Path) -> None:
    """Test running an experiment configs with `debug=fdr`.

    :param tmp_path: The temporary logging path.
    """
    command = ["dvc", "exp", "run", "-f", "-S", "debug=fdr"] + overrides
    run_sh_command(command)'''
