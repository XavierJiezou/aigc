from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
)
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPVisionModelWithProjection
import numpy as np
from torch import nn as nn
from src.models.components.attention_processor import AttnProcessor2_0FreeStyleNet as AttnProcessorFreeStyleNet
from src.models.components.attention_processor import AttnProcessor



class StableDiffusionFreestyleNetLitModule(LightningModule):

    def __init__(
        self,
        unet: UNet2DConditionModel | str,
        diffusion_schedule: DDPMScheduler | str,
        vae: AutoencoderKL | str,
        text_encoder: CLIPTextModel | str,
        optimizer: torch.optim.Optimizer,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "unet",
                "diffusion_schedule",
                "vae",
                "text_encoder",
            ],
        )

        self.unet = unet
        self.diffusion_schedule = diffusion_schedule
        self.vae = vae
        self.text_encoder = text_encoder

        if isinstance(unet, str):
            self.unet = UNet2DConditionModel.from_pretrained(unet, subfolder="unet")

        if isinstance(diffusion_schedule, str):
            self.diffusion_schedule = DDPMScheduler.from_pretrained(
                diffusion_schedule, subfolder="scheduler"
            )

        if isinstance(vae, str):
            self.vae = AutoencoderKL.from_pretrained(vae, subfolder="vae")

        if isinstance(text_encoder, str):
            self.text_encoder = CLIPTextModel.from_pretrained(
                text_encoder, subfolder="text_encoder"
            )
        
        self.unet = self.process_unet(self.unet)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # for tracking best so far validation accuracy
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def process_unet(self, unet: UNet2DConditionModel):
        unet.requires_grad_(False)
        # init adapter modules
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = AttnProcessorFreeStyleNet()
        unet.set_attn_processor(attn_procs)
        for name,p in unet.named_parameters():
            if "attn2.to_k" in name or "attn2.to_v" in name:
                p.requires_grad_(True)
        return unet

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
        with torch.no_grad():
            latents = self.vae.encode(
                batch["instance_images"], return_dict=True
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.diffusion_schedule.config.num_train_timesteps,
            (bs,),
            device=latents.device,
        ).long()
        noised_latents = self.diffusion_schedule.add_noise(latents, noise, timesteps)
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(batch["instance_prompt_ids"])[
                0
            ]  # encoder_hidden_states shape : bs 8 768
        

        noise_pred = self.unet.forward(
            noised_latents, timesteps, encoder_hidden_states,cross_attention_kwargs=dict(mask_label=batch["mask"])
        ).sample
        loss = F.mse_loss(noise_pred, noise, reduction="mean")

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
        return {"optimizer": optimizer}

if __name__ == "__main__":
    model = StableDiffusionFreestyleNetLitModule(
        unet="checkpoints/stablev15",
        diffusion_schedule="checkpoints/stablev15",
        vae="checkpoints/stablev15",
        text_encoder="checkpoints/stablev15",
        optimizer=torch.optim.Adam,
        compile=False,
    )
    batch = {
        "instance_prompt_ids": torch.randint(0, 256, (1, 77)),
        "instance_images": torch.randn(1, 3, 512, 512),
        "mask": torch.randint(0, 19, (1, 512, 512)),
        "drop_image_embeds": [1],
    }
    model.model_step(batch)
