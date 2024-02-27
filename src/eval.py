from typing import Any, Dict, List, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rich import get_console

from src.utils.logging.log_hyperparameters import log_hyperparameters
from src.utils.logging.ranked_logger import RankedLogger
from src.utils.misc.task_wrapper import task_wrapper
from src.utils.training_helpers.apply_extra_utilities import apply_extra_utilities
from src.utils.training_helpers.hydra_instantiators import instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)

console = get_console()


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    console.rule("[purple]INSTANTIATNG EVALUATION", style="white")
    assert cfg.paths.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.dataset._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("loggers"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": loggers,
        "trainer": trainer,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    console.rule("[purple]STARTING TESTING", style="white")
    log.info("Starting testing!")
    ckpt_path = cfg.paths.ckpt_path + "/best_model.ckpt"
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../config", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    apply_extra_utilities(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
