from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rich import get_console
from ruamel.yaml import YAML

from src.utils.logging.base_logger import BaseLogger
from src.utils.logging.log_hyperparameters import log_hyperparameters
from src.utils.misc.task_wrapper import task_wrapper
from src.utils.training_helpers.apply_extra_utilities import apply_extra_utilities
from src.utils.training_helpers.create_directories import create_directories
from src.utils.training_helpers.get_tracked_metric_value import get_tracked_metric_value
from src.utils.training_helpers.hydra_instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
)
from src.utils.training_helpers.merge_test_metrics import merge_test_metrics

log = BaseLogger(__name__)

console = get_console()


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    console.rule("[purple]INSTANTIATNG TRAINING", style="white")
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.dataset._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("loggers"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "loggers": loggers,
        "trainer": trainer,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        console.rule("[purple]STARTING TRAINING", style="white")
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("paths.ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        console.rule("[purple]STARTING TESTING", style="white")
        log.info("Starting testing!")
        # Try load best checkpoint from previous training step
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
        metrics_path = cfg.paths.get("dvclive_path") + "/metrics.json"
        new_metrics_overview_path = cfg.paths.get("dvclive_path") + "/plots/metrics/test/metrics_overview.csv"
        merge_test_metrics(metrics_path, new_metrics_overview_path)

    test_metrics = trainer.callback_metrics

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


# @hydra.main(version_base="1.3", config_path="../config", config_name="train.yaml")
def main() -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # Indicate experiment start
    console.rule("", style="green_yellow")
    console.rule("[purple]STARTED EXPERIMENT", style="green_yellow")
    console.rule("", style="green_yellow")

    # Load config
    cfg = DictConfig(YAML(typ="safe").load(open("params.yaml")))

    # Make sure output folder structure exists
    create_directories([cfg.paths.get("ckpt_path"), cfg.paths.get("log_path")])

    # Apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    apply_extra_utilities(cfg)

    # Train the model
    metric_dict, _ = train(cfg)

    # Safely retrieve optimized metric value for hydra-based hyperparameter optimization
    metric_value = get_tracked_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    return metric_value


if __name__ == "__main__":
    # Run the training
    main()
