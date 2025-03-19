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
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, PretrainedConfig


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


class HicoControlnetLitModule(LightningModule):

    def __init__(
        self,
        pretrained_model_name_or_path="checkpoints/stablev15",
        optimizer: torch.optim.Optimizer = None,
        fuse_type="avg",
        lr_scheduler: str = None,
        lr_num_cycles=1,
        lr_power=1.0,
        compile: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        )

        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )

        self.controlnet = ControlNetModel.from_unet(self.unet, conditioning_channels=1)

        self.diffusion_schedule = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.froze()
        self.controlnet.requires_grad_(True)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )

    def froze(self):
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

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

    def reshape_tensor(self, x: torch.Tensor):
        shape = x.shape
        bs = shape[0] * shape[1]
        if len(shape) == 2:
            return x.reshape(bs)
        x = x.reshape(bs, *shape[2:])
        return x

    def get_res_sample(self, latents: torch.Tensor, timesteps: torch.Tensor, batch):
        bs = latents.shape[0]
        with torch.no_grad():
            cls_text = self.reshape_tensor(batch["cls_text"])  # [bs*19,77]
            controlnet_encoder_hidden_states = self.text_encoder(cls_text)[0] # [bs*19,77,768]

        expanded_latents = latents.unsqueeze(1).repeat(
            1, 19, 1, 1, 1
        )  # [2, 19, 4, 64, 64]
        expanded_timesteps = timesteps.unsqueeze(1).repeat(1, 19)  # [2, 19]
        controlnet_image = batch["cls_mask"]

        expanded_latents = self.reshape_tensor(expanded_latents)
        expanded_timesteps = self.reshape_tensor(expanded_timesteps)
        # controlnet_encoder_hidden_states = self.reshape_tensor(
        #     controlnet_encoder_hidden_states
        # )
        controlnet_image = self.reshape_tensor(controlnet_image)
        controlnet_image = controlnet_image.unsqueeze(1).float()
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            expanded_latents,  # [bs*19,4,64,64]
            expanded_timesteps,  # [bs*19]
            encoder_hidden_states=controlnet_encoder_hidden_states,  # [bs*19, 77, 768]
            controlnet_cond=controlnet_image,  # [bs*19, 1, 512, 512]
            return_dict=False,
        )
        reshape_down_block_res_samples = []
        for res_sample in down_block_res_samples:
            shape = res_sample.shape[1:]
            res_sample = res_sample.reshape(bs, -1, *shape)
            if self.hparams.fuse_type == "avg":
                res_sample = torch.mean(res_sample,dim=1)
            elif self.hparams.fuse_type == "sum":
                res_sample = torch.sum(res_sample,dim=1)
            else:
                raise ValueError(f"Unsupported fuse_type '{self.hparams.fuse_type}', expected 'avg' or 'sum'.")
            reshape_down_block_res_samples.append(res_sample)

        mid_block_res_sample = mid_block_res_sample.reshape(
            bs, -1, *mid_block_res_sample.shape[1:]
        )
        if self.hparams.fuse_type == "avg":
            mid_block_res_sample = torch.mean(mid_block_res_sample,dim=1)
        elif self.hparams.fuse_type == "sum":
            mid_block_res_sample = torch.sum(mid_block_res_sample,dim=1)
        else:
            raise ValueError(f"Unsupported fuse_type '{self.hparams.fuse_type}', expected 'avg' or 'sum'.")

        return reshape_down_block_res_samples, mid_block_res_sample

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
            encoder_hidden_states = self.text_encoder(batch["instance_prompt_ids"])[
                0
            ]  # encoder_hidden_states shape : bs 8 768

        # print("noised_latents:",noised_latents.shape,noised_latents.dtype)
        # print("timesteps:",timesteps.shape,timesteps.dtype)
        # print("encoder_hidden_states:",encoder_hidden_states.shape,encoder_hidden_states.dtype)
        # print("controlnet_image:",controlnet_image.shape,controlnet_image.dtype)
        #

        down_block_res_samples, mid_block_res_sample = self.get_res_sample(
            latents=noised_latents, timesteps=timesteps, batch=batch
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
        trainable_params = self.controlnet.parameters()
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
    model = HicoControlnetLitModule()
    batch = {
        "instance_images": torch.randn(2, 3, 512, 512),
        "instance_prompt_ids": torch.randint(0, 100, (2, 77)),
        "cls_mask": torch.randint(low=0, high=2, size=(2, 19, 512, 512)),
        "cls_text": torch.randint(0, 100, (2, 19, 77)),
    }
    model.model_step(batch=batch)
