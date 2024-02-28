import subprocess  # nosec

import pytest


def run_sh_command(command):
    """Execute shell command and raise an exception if it fails.

    Args:
        command (List[str]): List of shell commands.

    Raises:
        subprocess.CalledProcessError: If the command execution fails.
    """
    try:
        subprocess.run(command, check=True, shell=True)  # nosec
    except subprocess.CalledProcessError as e:
        stdout_decoded = ""
        stderr_decoded = ""
        if e.stdout is not None:
            stdout_decoded = e.stdout.decode()
        if e.stderr is not None:
            stderr_decoded = e.stderr.decode()

        reasons = [
            f"Command: {command}",
            f"Exit code: {e.returncode}",
            f"Stdout: {stdout_decoded}",
            f"Stderr: {stderr_decoded}",
        ]

        # Join all elements of the reasons list into a single string
        reasons_str = "\n".join(reasons)

        # Raise exception with the output of the failed command
        pytest.fail(msg=reasons_str)
