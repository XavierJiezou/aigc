from typing import Any, Dict, Tuple,List
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
from src.models.components.attention import CrossAttention
from facer.face_parsing import FaRLFaceParser

class IPAdapterLitModule(LightningModule):
    def __init__(
        self,
        unet: UNet2DConditionModel | str,
        diffusion_schedule: DDPMScheduler,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: str,
        text_encoder: CLIPTextModel | str,
        vae: AutoencoderKL | str,
        image_encoder: CLIPVisionModelWithProjection | str,
        image_proj_model: ImageProjModel,
        text_mask_attn: CrossAttention,
        mask_text_attn: CrossAttention,
        face_seg: FaRLFaceParser | None = None,
        loss_type: List[str] = [],
        lr_num_cycles=1,
        lr_power=1.0,
        froze_components=["vae", "text_encoder"],
        set_timesteps=1,
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
                "image_encoder",
                "image_proj_model",
                "text_mask_attn",
                "mask_text_attn",
            ],
        )
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.diffusion_schedule = diffusion_schedule
        self.image_encoder = image_encoder
        self.image_proj_model = image_proj_model
        self.text_mask_attn = text_mask_attn
        self.mask_text_attn = mask_text_attn
        self.face_seg = face_seg
        self.net_clip = clip.load("checkpoints/clip/ViT-B-32.pt")[0]
        self.timesteps = torch.tensor([999]).long()

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
        if isinstance(image_encoder, str):
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                image_encoder
            )
        if set_timesteps > 0:
            self.diffusion_schedule.set_timesteps(set_timesteps)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.froze()

    def do_mask_text_attn(self,mask_features:torch.Tensor,text_features:torch.Tensor): 
        mask_text = self.mask_text_attn.forward(x=mask_features,context=text_features) # [bs,4,768] -> [bs,4,768]
        text_mask = self.text_mask_attn.forward(x=text_features,context=mask_features) # [bs,77,768] -> [bs,77,768]

        encoder_hidden_states = torch.cat([mask_text,text_mask],dim=1)  # [bs,81,768]
        return encoder_hidden_states
    
    def calculate_loss(
        self, pred_image: torch.Tensor, mask: torch.Tensor, caption_tokens: torch.Tensor
    ):
        total_loss = 0
        if "mask_loss" in self.hparams.loss_type:
            mask = mask.long()
            total_loss += F.cross_entropy(self.face_seg.net(pred_image)[0], mask)
        if "text_loss" in self.hparams.loss_type:
            x_tgt_pred_renorm = pred_image * 0.5 + 0.5
            x_tgt_pred_renorm = F.interpolate(
                x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False
            )
            clipsim, _ = self.net_clip(x_tgt_pred_renorm, caption_tokens)
            total_loss += 1 - clipsim.mean() / 100
        return total_loss


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
        # timesteps = torch.randint(
        #     0,
        #     self.diffusion_schedule.config.num_train_timesteps,
        #     (bs,),
        #     device=latents.device,
        # ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noised_latents = self.diffusion_schedule.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        with torch.no_grad():
            text_features = self.text_encoder(batch["instance_prompt_ids"])[
                0
            ]  # encoder_hidden_states shape : bs 77 768

        with torch.no_grad():
            image_embeds = self.image_encoder(batch["clip_images"]).image_embeds # [bs,768]
        
        image_embeds_ = []
        for image_embed ,drop_image_embed in zip(image_embeds,batch["drop_image_embeds"]):
            if drop_image_embed == 1:
                image_embeds_.append(torch.zeros_like(image_embed))
            else:
                image_embeds_.append(image_embed)
        image_embeds = torch.stack(image_embeds_)
        image_features = self.image_proj_model.forward(image_embeds)
        encoder_hidden_states = self.do_mask_text_attn(mask_features=image_features,text_features=text_features)


        # Predict the noise residual
        noise_pred = self.unet.forward(
            noised_latents,
            timesteps,
            encoder_hidden_states,
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
        output_image = (
            self.vae.decode(latents / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)

        mse_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        total_loss = self.calculate_loss(output_image,batch['mask'],batch['instance_prompt_ids'])
        total_loss += mse_loss
        return total_loss

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
        self.froze()
        trainable_params = [
            p for p in self.trainer.model.parameters() if p.requires_grad
        ]
        optimizer = self.hparams.optimizer(params=trainable_params)

        lr_scheduler = get_scheduler(
            self.hparams.lr_scheduler,
            optimizer=optimizer,
            num_cycles=self.hparams.lr_num_cycles,
            power=self.hparams.lr_power,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


if __name__ == "__main__":
    pass
