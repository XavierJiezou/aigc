import lightning as L
import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
from src.pipeline.pipeline_sota_controlnet_sd15 import StableDiffusionControlNetPipeline


class TrainingCallback(L.Callback):
    def __init__(self, sample_interval=100):

        self.sample_interval = sample_interval

        self.total_steps = 0
        self.to_tensor = T.ToTensor()
        self.recode_cond = False

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx
    ):

        self.total_steps += 1
        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            self.generate_a_sample(
                trainer,
                pl_module,
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
    ):
        device = pl_module.device
        pipeline: StableDiffusionControlNetPipeline = (
            StableDiffusionControlNetPipeline.from_pretrained(
                pl_module.pretrained_model_name_or_path, controlnet=pl_module.controlnet
            ).to(device)
        )
        pipeline.unet = pl_module.unet
        image_path = "data/mmcelebahq/mask/27000.png"
        prompt = "She is wearing lipstick. She is attractive and has straight hair."
        mask_list = [self.to_tensor(Image.open(image_path).convert("RGB"))]
        mask = np.array(Image.open(image_path))
        for i in range(19):
            local_mask = np.zeros_like(mask)
            local_mask[mask == i] = 255

            local_mask_rgb = Image.fromarray(local_mask).convert("RGB")
            local_mask_tensor = self.to_tensor(local_mask_rgb)
            mask_list.append(local_mask_tensor)
        condition_img = torch.stack(mask_list, dim=0)
        condition_img = condition_img.unsqueeze(0).to(device)
        images = pipeline.__call__(
            prompt=prompt,
            image=condition_img,
            num_inference_steps=50,
            guidance_scale=7.5,
            moe=pl_module.moe_list,
            height=512,
            width=512,
        ).images
        tensorboard = pl_module.logger.experiment

        if not self.recode_cond:
            tensorboard.add_image(
                "prompt",
                self.to_tensor(Image.open(image_path).convert("RGB")),
                self.total_steps,
            )

        tensorboard.add_image(
            f"Generated Image/step_{self.total_steps}",
            self.to_tensor(images[0]),
            self.total_steps,
        )
