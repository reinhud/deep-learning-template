from pathlib import Path

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, open_dict
from rich.prompt import Prompt

from src.utils.logging.ranked_logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        log_path = cfg.paths.log_path
        # Create the directory if it doesn't exist
        tags_log_path = Path(log_path)
        tags_log_path.mkdir(parents=True, exist_ok=True)
        # Save tags to file
        with open(Path(log_path, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)
