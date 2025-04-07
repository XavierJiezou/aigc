from typing import Any, Dict, Tuple
from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
    ControlNetModel,
)
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from transformers import CLIPTextModel
from torch import nn as nn
from src.models.components.simple_moe import MOE


class SotaControlSd15LitModule(LightningModule):

    def __init__(
        self,
        pretrained_model_name_or_path="checkpoints/stablev15",
        lora_config=None,
        optimizer=None,
        compile: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["lora_config"])

        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        )
        self.controlnet = ControlNetModel.from_unet(self.unet, conditioning_channels=3)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.unet.requires_grad_(True)

        self.unet.add_adapter(lora_config)
        self.lora_layers = list(
            filter(lambda p: p.requires_grad, self.unet.parameters())
        )

        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )

        self.vae.requires_grad_(False)
        self.vae.eval()

        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )

        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        self.diffusion_schedule = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.block_channels = [
            320,
            320,
            320,
            320,
            640,
            640,
            640,
            1280,
            1280,
            1280,
            1280,
            1280,
            1280,
        ]
        self.moe_list = nn.ModuleList(
            [MOE(channel, 20) for channel in self.block_channels]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()

    def reshape_tensor(self, x: torch.Tensor):
        shape = x.shape
        bs = shape[0] * shape[1]
        if len(shape) == 2:
            return x.reshape(bs)
        x = x.reshape(bs, *shape[2:])
        return x

    def get_res_sample(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        batch,
    ):
        controlnet_image = batch["condition"]  # bs 20 3 512 512
        mask_num = controlnet_image.shape[1]
        bs = controlnet_image.shape[0]
        expanded_latents = latents.unsqueeze(1).repeat(
            1, mask_num, 1, 1, 1
        )  # [bs, 20, 4, 64, 64]
        expanded_encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(
            1, mask_num, 1, 1
        )  # [bs, 20, 77, 768]
        expanded_timesteps = timesteps.unsqueeze(1).repeat(1, mask_num)  # [bs, 20]

        expanded_latents = self.reshape_tensor(expanded_latents)
        expanded_timesteps = self.reshape_tensor(expanded_timesteps)
        expanded_encoder_hidden_states = self.reshape_tensor(
            expanded_encoder_hidden_states
        )
        # controlnet_encoder_hidden_states = self.reshape_tensor(
        #     controlnet_encoder_hidden_states
        # )
        controlnet_image = self.reshape_tensor(controlnet_image)

        down_block_res_samples, mid_block_res_sample = self.controlnet.forward(
            expanded_latents,  # [bs*20,4,64,64]
            expanded_timesteps,  # [bs*20]
            encoder_hidden_states=expanded_encoder_hidden_states,  # [bs*20, 77, 768]
            controlnet_cond=controlnet_image,  # [bs*20, 3, 512, 512]
            return_dict=False,
        )
        # down_block_res_samples: List[torch.FloatTensor]
        # torch.Size([40, 320, 64, 64])
        # torch.Size([40, 320, 64, 64])
        # torch.Size([40, 320, 64, 64])
        # torch.Size([40, 320, 32, 32])
        # torch.Size([40, 640, 32, 32])
        # torch.Size([40, 640, 32, 32])
        # torch.Size([40, 640, 16, 16])
        # torch.Size([40, 1280, 16, 16])
        # torch.Size([40, 1280, 16, 16])
        # torch.Size([40, 1280, 8, 8])
        # torch.Size([40, 1280, 8, 8])
        # torch.Size([40, 1280, 8, 8])
        # mid_block_res_sample: torch.Size([40, 1280, 8, 8])

        down_block_res_samples_list = []
        for i in range(len(down_block_res_samples)):
            res_samples = down_block_res_samples[i]
            res_samples = res_samples.reshape(
                bs,
                mask_num,
                res_samples.shape[1],
                res_samples.shape[2],
                res_samples.shape[3],
            )
            res_samples = self.moe_list[i](res_samples)
            down_block_res_samples_list.append(res_samples)
        mid_block_res_sample_expert = mid_block_res_sample.reshape(
            bs,
            mask_num,
            mid_block_res_sample.shape[1],
            mid_block_res_sample.shape[2],
            mid_block_res_sample.shape[3],
        )
        mid_block_res_sample_expert = self.moe_list[-1](mid_block_res_sample_expert)
        return down_block_res_samples_list, mid_block_res_sample_expert

    def model_step(self, batch):
        # Convert images to latent space
        with torch.no_grad():
            latents = self.vae.encode(
                batch["image"], return_dict=True
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
            encoder_hidden_states = self.text_encoder(batch["instance_prompt_ids"])[
                0
            ]  # encoder_hidden_states shape : bs 8 768

        # down_block_res_samples, mid_block_res_sample = self.controlnet(
        #     noised_latents,
        #     timesteps,
        #     encoder_hidden_states=encoder_hidden_states,
        #     controlnet_cond=controlnet_image,
        #     return_dict=False,
        # )
        down_block_res_samples, mid_block_res_sample = self.get_res_sample(
            latents=noised_latents,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            batch=batch,
        )

        # Predict the noise residual
        noise_pred = self.unet.forward(
            noised_latents,
            timesteps,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        # Get the target for loss depending on the prediction type
        if self.diffusion_schedule.config.prediction_type == "epsilon":
            target = noise
        elif self.diffusion_schedule.config.prediction_type == "v_prediction":
            target = self.diffusion_schedule.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.diffusion_schedule.config.prediction_type}"
            )
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
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
        self.unet.requires_grad_(False)
        self.trainable_params = (
            self.lora_layers
            + [p for p in self.controlnet.parameters()]
            + [p for p in self.moe_list.parameters()]
        )
        for p in self.trainable_params:
            p.requires_grad_(True)
        optimizer = self.hparams.optimizer(params=self.trainable_params)

        return {
            "optimizer": optimizer,
        }


if __name__ == "__main__":
    from src.data.global_local_mmcelebahq_datamodule import (
        GlobalLocalMaskMMCelebaqHQDataModule,
    )

    data_module = GlobalLocalMaskMMCelebaqHQDataModule(
        root="data/mmcelebahq",
        batch_size=2,
        num_workers=0,
        drop_text_prob=0.5,
        drop_image_prob=0.5,
    )
    data_module.setup()
    train_dataloader = data_module.train_dataloader()

    from peft import LoraConfig
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    model = SotaControlSd15LitModule(
        pretrained_model_name_or_path="checkpoints/stablev15",
        lr_num_cycles=1,
        lr_power=1.0,
        compile=False,
        lora_config=lora_config
    )
    for batch in train_dataloader:
        break

    model.model_step(batch)
