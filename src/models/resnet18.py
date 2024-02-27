from typing import Any, Dict, Tuple

import lightning as L
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class LResNet18(L.LightningModule):
    """ResNet18 model for image classification."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        # TODO: make this dynamic
        output_path: str,  # Path to save the model checkpoints
        num_target_classes: int = 10,
    ):
        super().__init__()

        self.num_target_classes = num_target_classes
        self.output_path = output_path

        # Log hyperparameters
        self.save_hyperparameters()

        # Initialize loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Setup metrics
        self._setup_metrics()

        # Initialize a pretrained ResNet
        backbone = models.resnet18(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # Initialize prediction head
        self.classifier = nn.Linear(num_filters, self.num_target_classes)

    def _setup_metrics(self) -> None:
        """Setup metrics for train, val, and test."""
        metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=self.num_target_classes),
                "f1": MulticlassF1Score(num_classes=self.num_target_classes),
                "precision": MulticlassPrecision(num_classes=self.num_target_classes),
                "recall": MulticlassRecall(num_classes=self.num_target_classes),
            }
        )
        # Clone metrics for train, val, and test
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # For tracking best validation accuracy so far
        self.val_accuracy_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # By default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_accuracy_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        return y, logits, loss, preds

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        targets, logits, loss, preds = self.model_step(batch)

        # Update and log metrics
        self.train_loss(loss)
        self.train_metrics(preds, targets)
        self.log_dict(
            {
                "train_loss": self.train_loss,
                "train_epoch": int(self.current_epoch),
            },
            on_step=False,
            on_epoch=True,
        )
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        # Return loss for backpropagation
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        targets, logits, loss, preds = self.model_step(batch)

        # Update and log metrics
        self.val_loss(loss)
        self.val_metrics(preds, targets)
        self.log_dict(
            {
                "val_loss": self.val_loss,
                "val_epoch": int(self.current_epoch),
            },
            on_step=False,
            on_epoch=True,
        )
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # Log best validation accuracy so far
        accuracy = self.val_metrics["accuracy"].compute()  # get current val acc
        self.val_accuracy_best(accuracy)  # update best so far val acc
        # Log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val_accuracy_best", self.val_accuracy_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        targets, logits, loss, preds = self.model_step(batch)

        # Update and log metrics
        self.test_loss(loss)
        self.test_metrics(preds, targets)
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

        # Compute predicted probabilities
        predicted_probabilities = torch.softmax(logits, dim=1)

        # Log predictions, probabilities and targets
        for pred_class, prob, target in zip(preds, predicted_probabilities, targets):
            prob_pred_class = prob[pred_class.item()]  # Probability for predicted class
            self.log_dict(
                {
                    "test_probability_predicted_class": prob_pred_class,
                    "test_predictions": pred_class,
                    "test_targets": target,
                },
                on_step=True,
                on_epoch=False,
            )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers
                to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    '''def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm) -> None:
        """Configure the gradient clipping to control the exploding gradients problem during
        training. This method adjusts the gradient clipping value after a certain number of epochs.

        Gradient clipping is a technique to prevent the gradients from becoming too large, which can
        cause training instability. This method is called internally by Lightning during the backward
        pass to apply gradient clipping.

        :param optimizer: The optimizer for which gradient clipping needs to be configured.
        :param gradient_clip_val: The value for gradient clipping. If the current epoch is greater than 5,
                                    this value is doubled to adjust the clipping threshold.
        :param radient_clip_algorithm: The algorithm to use for gradient clipping,
                                        e.g., 'norm' for L2 norm clipping.

        Note:
            - Lightning handles the gradient clipping internally, so this method just configures
                the parameters for clipping.
            - The method modifies the gradient clipping value based on the current epoch to adapt
                the training process dynamically.
        """
        if self.current_epoch > 5 and gradient_clip_val:
            gradient_clip_val = gradient_clip_val * 2

        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm
        )'''
