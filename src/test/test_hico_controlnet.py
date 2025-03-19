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
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
    ControlNetModel,
)
from src.pipeline.pipeline_hico_controlnet import StableDiffusionHicoControlNetPipeline


class EvalDataset(Dataset):
    def __init__(self, root: str, mask_range=None,tokenizer_id="checkpoints/stablev15",):
        super().__init__()
        mask_range = mask_range or (27000, 30000)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            tokenizer_id,
            subfolder="tokenizer",
        )
        self.mask_range = [i for i in range(mask_range[0], mask_range[1])]
        self.root = root
        self.prompts = self.get_prompts()
        self.masks = self.get_masks()
        self.categories = [
            "background",
            "face",
            "nose",
            "eyeglasses", 
            "right eye",
            "left eye",
            "right eyebrow",
            "left eyebrow",
            "right ear",
            "left ear",
            "inner mouth",
            "upper lip",
            "lower lip",
            "hair",
            "hat",  
            "earring", 
            "necklace", #
            "neck",
            "clothing",
        ]

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
        mask = np.array(Image.open(mask_path))

        # 将类别信息记录下来
        cls_text = []
        cls_mask = []
        for i in range(len(self.categories)):
            tmp_mask = np.zeros_like(mask)
            tmp_mask[mask == i] = 1
            cls_mask.append(tmp_mask)
            # cls_text.append(self.categories[i])
            # 检查 tmp_mask 是否全为 0
            if np.any(tmp_mask):
                cls_text.append(self.categories[i])
            else:
                cls_text.append("")

        cls_mask = np.array(cls_mask)
        # cls_text = np.array(cls_text)
        
        cls_text = self.tokenizer(
            cls_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids


        return {
            "cls_mask": cls_mask,
            "cls_text": cls_text,
            "prompt": prompt,
            "filename": filename,
        }


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
        default="logs/train/runs/hico_controlnet/2025-03-17_11-31-04/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/hico_controlnet",
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
        default="checkpoints/stablev15",
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

    unet = UNet2DConditionModel.from_pretrained(args.base_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=1)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {}
    for k, v in ckpt.items():
        if k.startswith("controlnet."):
            state_dict[k[11:]] = v
    controlnet.load_state_dict(state_dict, strict=True)
    pipeline = StableDiffusionHicoControlNetPipeline.from_pretrained(
        args.base_model_path, controlnet=controlnet
    ).to(args.device)
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False

    return pipeline

def collect_fn(batchs):
    prompt = [batch["prompt"] for batch in batchs]
    filename = [batch["filename"] for batch in batchs]
    cls_mask = torch.stack([torch.tensor(batch["cls_mask"]) for batch in batchs])
    cls_text = torch.stack([batch["cls_text"] for batch in batchs])
    return {"prompt": prompt, "filename": filename, "cls_mask": cls_mask,"cls_text":cls_text}


def get_dataloader(root, mask_range, batch_size):
    dataset = EvalDataset(root=root, mask_range=mask_range)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collect_fn
    )
    return dataloader


def main():
    args = get_args()
    print("init pipeline...")
    pipeline: StableDiffusionHicoControlNetPipeline = get_pipeline(args)
    generator = torch.Generator().manual_seed(args.seed)

    print("init dataloader...")
    os.makedirs(args.output_dir, exist_ok=True)
    dataloader = get_dataloader(
        root=args.root,
        mask_range=args.mask_range,
        batch_size=args.batch_size,
    )
    for batch in dataloader:
        prompts = batch["prompt"]
        filenames = batch["filename"]
        cls_mask = batch["cls_mask"]
        cls_text = batch["cls_text"]


        images = pipeline.__call__(
            prompt=prompts,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            cls_text=cls_text,
            image=cls_mask,
        ).images
        # print(len(images))
        for image, filename in zip(images, filenames):
            save_path = os.path.join(args.output_dir, filename).replace(".png", ".jpg")
            image.save(save_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
