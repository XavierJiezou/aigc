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


class HicoLitModule(LightningModule):

    def __init__(
        self,
        pretrained_model_name_or_path="checkpoints/stablev15",
        optimizer: torch.optim.Optimizer = None,
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
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)

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
            unet_encoder_hidden_states = self.text_encoder(
                batch["instance_prompt_ids"]
            )[
                0
            ]  # encoder_hidden_states shape : bs 77 768
        bs_down_block_res_sample = []
        bs_mid_block_res_sample = []
        for cid in range(bs):
            # num_selected = min(obj_bbox.shape[0], self.max_objects_per_image)
            # bsid = batch_cond["num_selected"][cid]
            bsid = 19
            dot_noisy_latents = torch.repeat_interleave(
                torch.unsqueeze(noised_latents[cid], dim=0), repeats=bsid, dim=0
            )
            dot_timesteps = torch.repeat_interleave(
                torch.unsqueeze(timesteps[cid], dim=0), repeats=bsid, dim=0
            )
            encoder_hidden_states = self.text_encoder(batch["cls_text"][cid])[0]
            controlnet_image = batch["cls_mask"][cid].float()  # 19 512 512
            controlnet_image = controlnet_image.unsqueeze(1)  # 19 1 512 512

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                dot_noisy_latents,  # torch.Size([19, 4, 64, 64])
                dot_timesteps,  # torch.Size([19])
                encoder_hidden_states=encoder_hidden_states,  # torch.Size([19, 77, 768])
                controlnet_cond=controlnet_image,  # torch.Size([19, 1, 512, 512])
                return_dict=False,
            )

            ########## fuse - sum ##########
            # dot_down_block_res_sample = [torch.sum(down_block_res_samples[iii], dim=0) for iii in range(len(down_block_res_samples))]
            # dot_mid_block_res_sample = torch.sum(mid_block_res_sample, dim=0)

            ########## fuse - avg ##########
            dot_down_block_res_sample = [
                torch.sum(down_block_res_samples[iii], dim=0) / bsid
                for iii in range(len(down_block_res_samples))
            ]
            dot_mid_block_res_sample = torch.sum(mid_block_res_sample, dim=0) / bsid

            bs_down_block_res_sample.append(dot_down_block_res_sample)
            bs_mid_block_res_sample.append(dot_mid_block_res_sample)

        fus_down_block_res_samples = []
        fus_mid_block_res_sample = torch.stack(bs_mid_block_res_sample)
        for iix in range(len(bs_down_block_res_sample[0])):
            tmp_down = [
                bs_down_block_res_sample[iiy][iix]
                for iiy in range(len(bs_down_block_res_sample))
            ]
            fus_down_block_res_samples.append(torch.stack(tmp_down))

        del dot_down_block_res_sample
        del dot_mid_block_res_sample
        del bs_down_block_res_sample
        del bs_mid_block_res_sample

        # Predict the noise residual
        noise_pred = self.unet.forward(
            noised_latents,
            timesteps,
            unet_encoder_hidden_states,
            down_block_additional_residuals=[
                sample for sample in fus_down_block_res_samples
            ],
            mid_block_additional_residual=fus_mid_block_res_sample,
        ).sample
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

    def sample_images(self, batch, num_inference_steps=50, guidance_scale=0):
        with torch.no_grad():
            # Convert text prompt to embeddings
            text_embeddings = self.text_encoder(batch["instance_prompt_ids"])[0]
            bs = batch["instance_images"].shape[0]
            # Initialize random noise as the starting point
            latents = torch.randn((bs, 4, 64, 64), device=self.vae.device)

            # Prepare the time step schedule
            timesteps = torch.linspace(
                self.diffusion_schedule.config.num_train_timesteps - 1,
                0,
                num_inference_steps,
                dtype=torch.long,
                device=self.vae.device,
            )

            for t in timesteps:
                # Prepare the noisy latents input
                latent_model_input = latents

                # Scale the latent noise by the guidance scale
                # if guidance_scale > 1:
                #     latent_model_input = torch.cat([latent_model_input] * 2)

                bs_down_block_res_sample = []
                bs_mid_block_res_sample = []

                ### process mask
                for cid in range(bs):
                    # num_selected = min(obj_bbox.shape[0], self.max_objects_per_image)
                    # bsid = batch_cond["num_selected"][cid]
                    bsid = 19
                    dot_noisy_latents = torch.repeat_interleave(
                        torch.unsqueeze(latents[cid], dim=0),
                        repeats=bsid,
                        dim=0,
                    )
                    # dot_timesteps = torch.repeat_interleave(
                    #     torch.unsqueeze(timesteps[cid], dim=0), repeats=bsid, dim=0
                    # )
                    dot_timesteps = torch.repeat_interleave(
                        torch.unsqueeze(t, dim=0), repeats=bsid, dim=0
                    )
                    encoder_hidden_states = self.text_encoder(batch["cls_text"][cid])[0]
                    controlnet_image = batch["cls_mask"][cid].float()  # 19 512 512
                    controlnet_image = controlnet_image.unsqueeze(1)  # 19 1 512 512

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        dot_noisy_latents,  # torch.Size([19, 4, 64, 64])
                        dot_timesteps,  # torch.Size([19])
                        encoder_hidden_states=encoder_hidden_states,  # torch.Size([19, 77, 768])
                        controlnet_cond=controlnet_image,  # torch.Size([19, 1, 512, 512])
                        return_dict=False,
                    )

                    ########## fuse - sum ##########
                    # dot_down_block_res_sample = [torch.sum(down_block_res_samples[iii], dim=0) for iii in range(len(down_block_res_samples))]
                    # dot_mid_block_res_sample = torch.sum(mid_block_res_sample, dim=0)

                    ########## fuse - avg ##########
                    dot_down_block_res_sample = [
                        torch.sum(down_block_res_samples[iii], dim=0) / bsid
                        for iii in range(len(down_block_res_samples))
                    ]
                    dot_mid_block_res_sample = (
                        torch.sum(mid_block_res_sample, dim=0) / bsid
                    )

                    bs_down_block_res_sample.append(dot_down_block_res_sample)
                    bs_mid_block_res_sample.append(dot_mid_block_res_sample)

                fus_down_block_res_samples = []
                fus_mid_block_res_sample = torch.stack(bs_mid_block_res_sample)
                for iix in range(len(bs_down_block_res_sample[0])):
                    tmp_down = [
                        bs_down_block_res_sample[iiy][iix]
                        for iiy in range(len(bs_down_block_res_sample))
                    ]
                    fus_down_block_res_samples.append(torch.stack(tmp_down))

                del dot_down_block_res_sample
                del dot_mid_block_res_sample
                del bs_down_block_res_sample
                del bs_mid_block_res_sample

                # Predict the noise residual
                noise_pred = self.unet.forward(
                    latent_model_input,
                    t,
                    text_embeddings,
                    down_block_additional_residuals=[
                        sample for sample in fus_down_block_res_samples
                    ],
                    mid_block_additional_residual=fus_mid_block_res_sample,
                ).sample

                # Perform guidance
                if guidance_scale > 1:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.diffusion_schedule.step(noise_pred, t, latents,return_dict=False)[0]

        # Decode the final latent to the image space
        images = self.vae.decode(latents / self.vae.config.scaling_factor)["sample"]
        print(images.max(),images.min())
        images = self.image_processor.postprocess(images, output_type="pil", do_denormalize=[True] * bs)
            

        return images


if __name__ == "__main__":
    model = HicoLitModule()
    # batch = {
    #     "instance_images": torch.randn(2, 3, 512, 512),
    #     "instance_prompt_ids": torch.randint(0, 100, (2, 77)),
    #     "cls_mask": torch.randint(low=0, high=2, size=(2, 19, 512, 512)),
    #     "cls": torch.randint(0, 19, (2,)),
    #     "cls_text": torch.randint(0, 100, (2, 19, 77)),
    #     "mask": torch.randint(0, 19, (2, 512, 512)),
    # }
    # image = model.sample_images(batch,num_inference_steps=2)
    # print(image.shape)
    ckpt_path = torch.load("logs/train/runs/hico/2025-03-16_00-32-18/checkpoints/last.ckpt",map_location="cpu")["state_dict"]
    state_dict = {}
    for k,v in ckpt_path.items():
        # k = k[4:]
        state_dict[k] = v
    model.load_state_dict(state_dict)
    model.cuda()
    from src.data.hico_t2i_datamodule import HicoDataModule
    dataloader = HicoDataModule(batch_size=2)
    dataloader.setup()
    val_dataloader = dataloader.val_dataloader()
    for data in val_dataloader:
        data["instance_images"] = data["instance_images"].cuda()
        data["instance_prompt_ids"] = data["instance_prompt_ids"].cuda()
        data["cls_mask"] = data["cls_mask"].cuda()
        # data["cls"] = data["cls"].cuda()
        data["cls_text"] = data["cls_text"].cuda()
        data["mask"] = data["mask"].cuda()
        images = model.sample_images(data,num_inference_steps=50)
        images[0].save("0.png")
        images[1].save("1.png")
        break



