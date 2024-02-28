import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


class TestConfigurations:
    """Tests the Hydra configurations for training and evaluation."""

    def test_train_config(self, cfg_train: DictConfig) -> None:
        """Tests the training configuration provided by the `cfg_train` pytest fixture.

        :param cfg_train: A DictConfig containing a valid training configuration.
        """
        # Ensure that the configuration is not None
        assert cfg_train
        assert cfg_train.dataset
        assert cfg_train.model
        assert cfg_train.trainer

        HydraConfig().set_config(cfg_train)

        # Ensure that the configuration can be instantiated
        hydra.utils.instantiate(cfg_train.dataset)
        hydra.utils.instantiate(cfg_train.model)
        hydra.utils.instantiate(cfg_train.trainer)

    def test_eval_config(self, cfg_eval: DictConfig) -> None:
        """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

        :param cfg_train: A DictConfig containing a valid evaluation configuration.
        """
        # Ensure that the configuration is not None
        assert cfg_eval
        assert cfg_eval.dataset
        assert cfg_eval.model
        assert cfg_eval.trainer

        HydraConfig().set_config(cfg_eval)

        # Ensure that the configuration can be instantiated
        hydra.utils.instantiate(cfg_eval.dataset)
        hydra.utils.instantiate(cfg_eval.model)
        hydra.utils.instantiate(cfg_eval.trainer)
