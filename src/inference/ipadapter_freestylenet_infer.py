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
from src.models.ipadapter_freestylenet import StableDiffusionFreestyleNetLitModule


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="She is wearing lipstick. She is attractive and has straight hair.",
    )
    parser.add_argument("--mask", default="data/mmcelebahq/mask/27000.png")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/train/runs/ipadapter_freestylenet/2025-03-13_23-44-44/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--model_config", type=str, default="configs/model/ipadapter_freestylenet.yaml"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ipadapter_freestylenet/",
    )
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
    model: StableDiffusionFreestyleNetLitModule = hydra.utils.instantiate(model_config)
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

    return pipeline


def main():
    args = get_args()
    print("init pipeline...")
    pipeline: StableDiffusionPipeline = get_pipeline(args)
    print("start inference...")

    mask = np.array(Image.open(args.mask))
    mask = torch.tensor(mask).to(device=args.device).unsqueeze(0).long()
    os.makedirs(args.output_dir, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    image = pipeline.__call__(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        cross_attention_kwargs=dict(mask_label=mask),
        guidance_scale=0
    )
    # print(len(image),len(image[0])) 1 1
    image = image[0][0]
    base_name = os.path.basename(args.mask)
    output_path = os.path.join(args.output_dir, base_name).replace(".png", ".jpg")
    image.save(output_path)
    print(f"done.image saved to {output_path}")


if __name__ == "__main__":
    main()
