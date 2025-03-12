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
from src.models.components.attention_processor import (
    AttnProcessor2_0 as AttnProcessor,
    IPAttnProcessor2_0_pos as IPAttnProcessor,
)
import numpy as np
from torch import nn as nn


class SegEncoder(nn.Module):
    # 位置编码
    def __init__(self, num_classes=19, embedding_dim=32, cross_attention_dim=None):
        super().__init__()
        self.seg_embedding = nn.Embedding(num_classes, embedding_dim)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, stride=2, padding=1),  # 下采样
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, cross_attention_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(
                (16, 16)
            ),  # 输出形状: (batch_size, cross_attention_dim, 16, 16)
        )

    def forward(self, x):
        emb = self.seg_embedding(x)  # (batch_size, H, W, embedding_dim)
        emb = emb.permute(0, 3, 1, 2)  # (batch_size, embedding_dim, H, W)
        x = self.conv_layers(emb)
        batch_size, c, h, w = x.shape
        x = x.view(batch_size, c, -1).permute(0, 2, 1)  # 转换为 (batch_size, 256, 768)
        return x


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: float = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int] = (16, 16)) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        h, w, c = pe.shape
        pe = pe.reshape(h * w, c)
        return pe

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class IpadapterSegEncoderLitModule(LightningModule):

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
        self.seg_encoder = SegEncoder(
            embedding_dim=32, cross_attention_dim=self.unet.config.cross_attention_dim
        )
        self.pe_layer = PositionEmbeddingRandom(
            self.unet.config.cross_attention_dim // 2
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
        unet_sd = unet.state_dict()
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
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=256,
                    pos=self.pe_layer.forward()
                )
                attn_procs[name].load_state_dict(weights)
        unet.set_attn_processor(attn_procs)
        # for name,params in unet.named_parameters():
        #     if "attn2" in name:
        #         params.requires_grad = True
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
        mask_encoder_hidden_states = self.seg_encoder(batch["mask"]) # mask_encoder_hidden_states shape : bs 256 768

        # 随机drop mask
        image_embeds_ = []
        for image_embed, drop_image_embed in zip(
            mask_encoder_hidden_states, batch["drop_image_embeds"]
        ):
            if drop_image_embed == 1:
                image_embeds_.append(torch.zeros_like(image_embed))
            else:
                image_embeds_.append(image_embed)
        mask_encoder_hidden_states = torch.stack(image_embeds_)

        encoder_hidden_states = torch.cat([encoder_hidden_states, mask_encoder_hidden_states], dim=1) # bs 264 768
        noise_text_pred = self.unet.forward(
            noised_latents, timesteps, encoder_hidden_states
        ).sample
        loss = F.mse_loss(noise_text_pred, noise, reduction="mean")

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
    model = IpadapterSegEncoderLitModule(
        unet="checkpoints/stablev15",
        diffusion_schedule="checkpoints/stablev15",
        vae="checkpoints/stablev15",
        text_encoder="checkpoints/stablev15",
        optimizer=torch.optim.Adam,
        compile=False,
    )
    total = 0
    for name,p in model.named_parameters():
        if p.requires_grad:
            print(name)
            total += p.numel()
    print(total / 1e6)
