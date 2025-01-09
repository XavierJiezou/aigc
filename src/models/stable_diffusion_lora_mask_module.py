from typing import Any, Dict, Tuple, List
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from transformers import CLIPFeatureExtractor, CLIPTextModel
from peft import LoraConfig
from torch import nn as nn
from src.models.components.mask_encoder_spi import MaskEncoder
from src.models.components.proj import Proj


class StableDiffusionLoraMaskLitModule(LightningModule):

    def __init__(
        self,
        unet: UNet2DConditionModel | str,
        diffusion_schedule: DDPMScheduler,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        text_encoder: CLIPTextModel | str,
        vae: AutoencoderKL | str,
        mask_encoder: MaskEncoder | str,
        mask_encoder_weight_path: str,
        projection: Proj,
        lora_config: LoraConfig = None,
        froze_components: List[str] = [],
        compile: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "unet",
                "diffusion_schedule",
                "text_encoder",
                "vae",
                "lora_config",
                "mask_encoder",
                "projection",
            ],
        )
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.diffusion_schedule = diffusion_schedule
        self.mask_encoder = mask_encoder
        self.lora_config: LoraConfig = lora_config
        self.projection = projection

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

        if lora_config is not None:
            self.unet.add_adapter(lora_config)
            # for p in self.unet.parameters():
            #     p.requires_grad = True

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.load_mask_encoder_weight(mask_encoder_weight_path)

    def load_mask_encoder_weight(self, weight_path: str):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """

        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.froze()

    def model_step(self, batch):
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
        mask_embedding = self.mask_encoder.forward(batch["mask"])
        # print(mask_embedding.shape,encoder_hidden_states.shape)
        encoder_hidden_states = torch.cat([encoder_hidden_states, mask_embedding],dim=1)
        encoder_hidden_states = self.projection.forward(encoder_hidden_states)
        noise_pred = self.unet.forward(
            noised_latents, timesteps, encoder_hidden_states
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
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
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
        params = [p for p in self.trainer.model.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer(params=params)
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


if __name__ == "__main__":
    pass
