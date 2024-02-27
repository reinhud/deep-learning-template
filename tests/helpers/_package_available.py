import importlib.metadata
import platform

from lightning.fabric.accelerators import TPUAccelerator


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    :param package_name: The name of the package to be checked.

    :return: `True` if the package is available. `False` otherwise.
    """
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


_TPU_AVAILABLE = TPUAccelerator.is_available()

_IS_WINDOWS = platform.system() == "Windows"

_DEEPSPEED_AVAILABLE = not _IS_WINDOWS and _package_available("deepspeed")
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _package_available("fairscale")
