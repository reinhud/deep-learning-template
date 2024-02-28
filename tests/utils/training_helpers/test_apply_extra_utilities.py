from omegaconf import DictConfig, OmegaConf

from src.utils.training_helpers.apply_extra_utilities import apply_extra_utilities


class TestApplyExtraUtilities:

    def test_apply_extra_utilities_no_extras_config(self, cfg_train: DictConfig):
        cfg = OmegaConf.create({})
        apply_extra_utilities(cfg)
        # Add assertions to check if the function behaves as expected when no extras config is provided

    def test_apply_extra_utilities_ignore_warnings(self, cfg_train: DictConfig):
        cfg = OmegaConf.create({"extras": {"ignore_warnings": True}})
        apply_extra_utilities(cfg)
        # Add assertions to check if the function behaves as expected when ignore_warnings is set to True

    """def test_apply_extra_utilities_enforce_tags(self, cfg_train: DictConfig):
        cfg = OmegaConf.create({"extras": {"enforce_tags": True}})
        # Mock the enforce_tags function or provide a suitable test case for it
        apply_extra_utilities(cfg)
        # Add assertions to check if the function behaves as expected when enforce_tags is set to True

    def test_apply_extra_utilities_print_config(self, cfg_train: DictConfig):
        cfg = OmegaConf.create({"extras": {"print_config": True}})
        # Mock the print_config_tree function or provide a suitable test case for it
        apply_extra_utilities(cfg)
        # Add assertions to check if the function behaves as expected when print_config is set to True"""
