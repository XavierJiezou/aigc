import warnings

warnings.filterwarnings("ignore")
import argparse
import torch
import os
from src.models.controlnet_module import ControlLitModule
import hydra
from PIL import Image
import json
import numpy as np
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPFeatureExtractor
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader, Dataset
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)
from diffusers.utils import load_image


class EvalDataset(Dataset):
    def __init__(self, root: str, mask_range=None):
        super().__init__()
        mask_range = mask_range or (27000, 30000)
        self.mask_range = [i for i in range(mask_range[0], mask_range[1])]
        self.root = root
        self.prompts = self.get_prompts()
        self.masks = self.get_masks()

    def get_masks(self):
        masks = []
        for i in self.mask_range:
            mask_path = os.path.join(self.root, "mask", f"{i}.png")
            masks.append(mask_path)
        return masks

    def get_prompts(self):
        prompts = []
        for i in self.mask_range:
            text_path = os.path.join(self.root, "text", f"{i}.txt")
            with open(text_path, mode="r") as f:
                prompt = f.readlines()[0]
                prompts.append(prompt)
        return prompts

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        prompt = self.prompts[index]
        mask_path = self.masks[index]
        filename = os.path.basename(mask_path)
        mask = load_image(mask_path)
        return {"prompt": prompt, "filename": filename, "mask": mask}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/mmcelebahq",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/train/runs/sdxl_t2i_module/2025-03-14_11-32-12/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--model_config", type=str, default="configs/model/sdxl_t2i_module.yaml"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/sdxl_t2i_module",
    )
    parser.add_argument(
        "--mask_range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=[27000, 30000],
        help="Range of mask indices to process",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="checkpoints/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance_scale", type=int, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    return args


def get_pipeline(args):

    adapter = T2IAdapter(
        in_channels=3,
        channels=(320, 640, 1280, 1280),
        num_res_blocks=2,
        downscale_factor=16,
        adapter_type="full_adapter_xl",
    )
    weights = torch.load(args.ckpt_path)["state_dict"]
    adapter_weights = {}
    for k, v in weights.items():
        if k.startswith("t2iadapter.adapter"):
            adapter_weights[k[11:]] = v
    adapter.load_state_dict(adapter_weights)

    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        args.base_model_path, adapter=adapter
    )
    # pipe.scheduler = EulerAncestralDiscreteSchedulerTest.from_config(pipe.scheduler.config)
    return pipe

def collect_fn(batchs):
    prompt = [batch["prompt"] for batch in batchs]
    filename = [batch["filename"] for batch in batchs]
    masks = [batch["mask"] for batch in batchs]
    return {"prompt": prompt, "filename": filename, "mask": masks}


def get_dataloader(root, mask_range, height, width, batch_size):
    dataset = EvalDataset(root=root, mask_range=mask_range)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collect_fn
    )
    return dataloader


def main():
    args = get_args()
    print("init pipeline...")
    pipeline: StableDiffusionPipeline = get_pipeline(args)
    generator = torch.Generator().manual_seed(args.seed)

    print("init dataloader...")
    os.makedirs(args.output_dir, exist_ok=True)
    dataloader = get_dataloader(
        root=args.root,
        mask_range=args.mask_range,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
    )
    for batch in dataloader:
        prompts = batch["prompt"]
        filenames = batch["filename"]
        mask = batch["mask"]

        images = pipeline.__call__(
            prompt=prompts,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            image=mask
        ).images
        for image, filename in zip(images, filenames):
            save_path = os.path.join(args.output_dir, filename).replace(".png", ".jpg")
            image.save(save_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
