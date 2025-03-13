from typing import Any, Dict, Tuple

import torch
from diffusers.optimization import get_scheduler
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    EulerDiscreteScheduler,
    T2IAdapter,
)
import torch.nn.functional as F
import numpy as np
from torch import nn as nn


class StableDiffusionXLT2ILitModule(LightningModule):

    def __init__(
        self,
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        revision=None,
        variant=None,
        pretrained_vae_model_name_or_path=None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler="constant",
        lr_warmup_steps=500,
        lr_num_cycles=1,
        lr_power=1.0,
        max_train_steps=100000,
        compile: bool = False,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        vae_path = (
            pretrained_model_name_or_path
            if pretrained_vae_model_name_or_path is None
            else pretrained_vae_model_name_or_path
        )
        self.vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if pretrained_vae_model_name_or_path is None else None,
            revision=revision,
            variant=variant,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
            variant=variant,
        )
        self.t2iadapter = T2IAdapter(
            in_channels=3,
            channels=(320, 640, 1280, 1280),
            num_res_blocks=2,
            downscale_factor=16,
            adapter_type="full_adapter_xl",
        )

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        # for tracking best so far validation accuracy
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.train_loss.reset()
        self.test_loss.reset()

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

        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32, device="cuda"):
            sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
            schedule_timesteps = self.noise_scheduler.timesteps.to(device)
            timesteps = timesteps.to(device)

            step_indices = [
                (schedule_timesteps == t).nonzero().item() for t in timesteps
            ]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        pixel_values = batch["pixel_values"]
        # encode pixel values with batch size of at most 8 to avoid OOM
        latents = []
        for i in range(0, pixel_values.shape[0], 8):
            latents.append(
                self.vae.encode(pixel_values[i : i + 8]).latent_dist.sample()
            )
        latents = torch.cat(latents, dim=0)
        latents = latents * self.vae.config.scaling_factor
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Cubic sampling to sample a random timestep for each image.
        # For more details about why cubic sampling is used, refer to section 3.4 of https://arxiv.org/abs/2302.08453
        timesteps = torch.rand((bsz,), device=latents.device)
        timesteps = (1 - timesteps**3) * self.noise_scheduler.config.num_train_timesteps
        timesteps = timesteps.long().to(self.noise_scheduler.timesteps.dtype)
        timesteps = timesteps.clamp(
            0, self.noise_scheduler.config.num_train_timesteps - 1
        )

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # Scale the noisy latents for the UNet
        sigmas = get_sigmas(
            timesteps,
            len(noisy_latents.shape),
            noisy_latents.dtype,
            noisy_latents.device,
        )
        inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

        # Adapter conditioning.
        t2iadapter_image = batch["conditioning_pixel_values"]
        down_block_additional_residuals = self.t2iadapter(t2iadapter_image)
        down_block_additional_residuals = [
            sample for sample in down_block_additional_residuals
        ]
        # Predict the noise residual
        model_pred = self.unet(
            inp_noisy_latents,
            timesteps,
            encoder_hidden_states=batch["prompt_ids"],
            added_cond_kwargs=batch["unet_added_conditions"],
            down_block_additional_residuals=down_block_additional_residuals,
            return_dict=False,
        )[0]

        # Denoise the latents
        denoised_latents = model_pred * (-sigmas) + noisy_latents
        weighing = sigmas**-2.0

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = latents  # we are computing loss against denoise latents
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        # MSE loss
        loss = torch.mean(
            (
                weighing.float() * (denoised_latents.float() - target.float()) ** 2
            ).reshape(target.shape[0], -1),
            dim=1,
        )
        loss = loss.mean()

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)

        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        trainable_params = [
            p for p in self.trainer.model.parameters() if p.requires_grad
        ]
        optimizer = self.hparams.optimizer(params=trainable_params)
        lr_scheduler = get_scheduler(
            self.hparams.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.hparams.lr_warmup_steps,
            num_training_steps=self.hparams.max_train_steps,
            num_cycles=self.hparams.lr_num_cycles,
            power=self.hparams.lr_power,
        )
        return {"optimizer": optimizer}
