from typing import Any, Dict, Tuple, List
from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
)
import torch
import clip
from torch import nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from transformers import CLIPTextModel, CLIPVisionModelWithProjection
from src.models.components.image_proj_model import ImageProjModel
from diffusers.optimization import get_scheduler
from src.models.components.mask_encoder_spi import MaskEncoder
from src.models.components.proj import Proj


class IPAdapterMaskSpiProjLitModule(LightningModule):
    def __init__(
        self,
        unet: UNet2DConditionModel | str,
        diffusion_schedule: DDPMScheduler,
        optimizer: torch.optim.Optimizer,
        text_encoder: CLIPTextModel | str,
        vae: AutoencoderKL | str,
        mask_encoder: MaskEncoder | str,
        mask_encoder_weight_path: str,
        froze_components=["vae", "text_encoder"],
        mask_proj: Proj | nn.Identity = nn.Identity(),
        text_proj: Proj | nn.Identity = nn.Identity(),
        text_crossattention=True,
        compile: bool = False,
    ) -> None:
        super().__init__()

        if text_crossattention:
            from src.models.components.attention_train_two_processor import (
                AttnProcessor2_0 as AttnProcessor,
                IPAttnProcessor2_0 as IPAttnProcessor,
            )
        else:
            from src.models.components.attention_processor import (
                AttnProcessor2_0 as AttnProcessor,
                IPAttnProcessor2_0 as IPAttnProcessor,
            )

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "unet",
                "diffusion_schedule",
                "text_encoder",
                "vae",
                "mask_encoder",
                "mask_proj",
                "text_proj",
            ],
        )
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.diffusion_schedule = diffusion_schedule
        self.mask_encoder = mask_encoder

        if isinstance(unet, str):
            self.unet = UNet2DConditionModel.from_pretrained(unet, subfolder="unet")

        if isinstance(vae, str):
            self.vae = AutoencoderKL.from_pretrained(vae, subfolder="vae")

        if isinstance(text_encoder, str):
            self.text_encoder = CLIPTextModel.from_pretrained(
                text_encoder, subfolder="text_encoder"
            )

        if isinstance(diffusion_schedule, str):
            self.diffusion_schedule = DDPMScheduler.from_pretrained(
                diffusion_schedule, subfolder="scheduler"
            )

        self.mask_proj = mask_proj

        self.text_proj = text_proj
        # init adapter modules
        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                if text_crossattention:
                    weights = {
                        "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                        "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                        "to_k_text.weight": unet_sd[layer_name + ".to_k.weight"],
                        "to_v_text.weight": unet_sd[layer_name + ".to_v.weight"],
                    }
                else:
                    weights = {
                        "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                        "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                    }
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=196,
                )
                attn_procs[name].load_state_dict(weights)
        self.unet.set_attn_processor(attn_procs)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.load_mask_encoder_weight(weight_path=mask_encoder_weight_path)
        self.froze()

    def load_mask_encoder_weight(self, weight_path):
        ckpt: Dict = torch.load(weight_path)["state_dict"]
        state_dict = {}
        for k, v in ckpt.items():
            if k[:13] == "mask_encoder.":
                new_k = k[13:]
                state_dict[new_k] = v
        self.mask_encoder.load_state_dict(state_dict, strict=True)

    def froze(self):
        for component in self.hparams.froze_components:
            component: nn.Module = getattr(self, component)
            component.eval()
            component.requires_grad_(False)
            for param in component.parameters():
                param.requires_grad = False
        if "unet" in self.hparams.froze_components:
            for proc in self.unet.attn_processors.values():
                for param in proc.parameters():
                    param.requires_grad = True

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
        self.froze()

    def model_step(self, batch):
        # Convert images to latent space
        with torch.no_grad():
            latents = self.vae.encode(
                batch["instance_images"], return_dict=True
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bs = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.diffusion_schedule.config.num_train_timesteps,
            (bs,),
            device=latents.device,
        ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noised_latents = self.diffusion_schedule.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        with torch.no_grad():
            text_features = self.text_encoder(batch["instance_prompt_ids"])[
                0
            ]  # encoder_hidden_states shape : bs 77 1024

        text_features = self.text_proj(text_features)

        with torch.no_grad():
            image_embeds = self.mask_encoder.forward(batch["mask"])  # [bs,196,768]

        image_embeds = self.mask_proj(image_embeds)

        image_embeds_ = []
        for image_embed, drop_image_embed in zip(
            image_embeds, batch["drop_image_embeds"]
        ):
            if drop_image_embed == 1:
                image_embeds_.append(torch.zeros_like(image_embed))
            else:
                image_embeds_.append(image_embed)
        image_embeds = torch.stack(image_embeds_)
        encoder_hidden_states = torch.cat((text_features, image_embeds), dim=1)

        # Predict the noise residual
        noise_pred = self.unet.forward(
            noised_latents,
            timesteps,
            encoder_hidden_states,
            return_dict=False,
        )[0]

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
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
            "train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
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

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.unet = torch.compile(self.unet)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        # self.froze()
        trainable_params = [
            p for p in self.trainer.model.parameters() if p.requires_grad
        ]
        optimizer = self.hparams.optimizer(params=trainable_params)
        return {
            "optimizer": optimizer,
        }


if __name__ == "__main__":
    pass
