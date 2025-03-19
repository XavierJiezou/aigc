import warnings

warnings.filterwarnings("ignore")
import shutil
import imageio
import argparse
import os
from transformers import CLIPTokenizer, CLIPFeatureExtractor
import hydra
from omegaconf import OmegaConf
from typing import List
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import CLIPImageProcessor
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
from diffusers import DPMSolverMultistepScheduler
from src.models.sd_seg_encoder_vit_pos_lora import StableDiffusionSegEncoderVitPosLoraLitModule
from mmengine.visualization import Visualizer


def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [
                torch.Generator(device).manual_seed(seed_item) for seed_item in seed
            ]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator


class IPAdapterPipeline:
    def __init__(
        self,
        sd_pipeline: StableDiffusionPipeline,
        ipadapter: StableDiffusionSegEncoderVitPosLoraLitModule,
        device: str,
        size=512,
    ):
        self.sd_pipeline: StableDiffusionPipeline = sd_pipeline
        self.ipadapter = ipadapter
        self.device = device

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                # pil_image = [pil_image]
                pil_image = np.array(pil_image).astype(np.int64)
            clip_image =  torch.tensor(pil_image).unsqueeze(0).to(self.device)
            clip_image_embeds = self.ipadapter.seg_encoder(clip_image)
            feature_map = clip_image_embeds[0].permute(1,0).reshape(768,16,16)
            overlaid_image = np.array(Image.open("data/mmcelebahq/face/27000.jpg"))
            image = Visualizer.draw_featmap(featmap=feature_map,overlaid_image=overlaid_image)
            Image.fromarray(image).save("feature_map.png")
            print(clip_image_embeds.shape)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device)
        image_prompt_embeds = clip_image_embeds
        uncond_image_prompt_embeds = self.ipadapter.seg_encoder(
            torch.zeros_like(clip_image)
        )
        return image_prompt_embeds, uncond_image_prompt_embeds
    

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=50,
        **kwargs,
    ):
        
        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            )

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(
            bs_embed * num_samples, seq_len, -1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
            1, num_samples, 1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
            bs_embed * num_samples, seq_len, -1
        )

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.sd_pipeline.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            # prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            # negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.sd_pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="She is wearing lipstick. She is attractive and has straight hair.")
    parser.add_argument("--mask", default="data/mmcelebahq/mask/27000.png")
    parser.add_argument("--ckpt_path", type=str, default="logs/train/runs/sd_seg_encoder_vit_pos_lora/2025-03-18_12-08-15/checkpoints/last.ckpt")
    parser.add_argument("--model_config", type=str, default="configs/model/sd_seg_encoder_vit_pos_lora.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/sd_seg_encoder_vit_pos_lora")
    parser.add_argument("--tokenizer_id", type=str, default="checkpoints/stablev15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance_scale", type=int, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    args = parser.parse_args()
    return args


def get_pipeline(args):
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model_config = OmegaConf.load(args.model_config)  # 加载model config file
    model: StableDiffusionSegEncoderVitPosLoraLitModule = hydra.utils.instantiate(model_config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained(
        args.tokenizer_id,
        subfolder="tokenizer",
    )
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "checkpoints/stablev15", subfolder="feature_extractor"
    )
    scheduler = model.diffusion_schedule
    pipeline = StableDiffusionPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        unet=model.unet,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=None,
    ).to(args.device)
    # pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    # pipeline.enable_xformers_memory_efficient_attention()
    ipadapter_pipe = IPAdapterPipeline(
        sd_pipeline=pipeline, ipadapter=model, device=args.device
    )
    return ipadapter_pipe



def main():
    args = get_args()
    print("init pipeline...")
    pipeline: IPAdapterPipeline = get_pipeline(args)
    print("start inference...")

    mask = Image.open(args.mask)
    print(args.prompt)
    mask.save("1.png")
    os.makedirs(args.output_dir, exist_ok=True)


    image = pipeline.generate(
        pil_image=mask,
        prompt=args.prompt,
        image=mask,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_samples=1,
    )[0]

    base_name = os.path.basename(args.mask)
    output_path = os.path.join(args.output_dir, base_name).replace(".png",".jpg")
    image.save(output_path)
    print(f"done.image saved to {output_path}")


if __name__ == "__main__":
    main()
