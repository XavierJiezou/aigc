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
from lightning import seed_everything


class EvalDataset(Dataset):
    def __init__(self, root: str, mask_range=None, height=512, width=512):
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
            text_path = os.path.join(self.root,"text",f"{i}.txt")
            with open(text_path,mode='r') as f:
                prompt = f.readlines()[0]
                prompts.append(prompt)
        return prompts

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        prompt = self.prompts[index]
        mask_path = self.masks[index]
        filename = os.path.basename(mask_path)
        prompt = prompt.strip()
        # return {"prompt": "This man has gray hair, and high cheekbones. He wears necktie.", "filename": f"{index}.png"}

        return {"prompt": prompt, "filename": filename}


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
        default="logs/train/runs/stable_diffusion_attn_only_text/2025-03-13_23-38-35/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--model_config", type=str, default="configs/model/stable_diffusion_attn_only_text.yaml"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/ipadapter_only_text",
    )
    parser.add_argument(
        "--mask_range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=[27000, 30000],
        help="Range of mask indices to process",
    )
    parser.add_argument("--tokenizer_id", type=str, default="checkpoints/stablev15")
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
    ckpt = None
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model_config = OmegaConf.load(args.model_config)  # 加载model config file
    model = hydra.utils.instantiate(model_config)
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
    tokenizer = CLIPTokenizer.from_pretrained(
        args.tokenizer_id,
        subfolder="tokenizer",
    )
    feature_extractor = CLIPFeatureExtractor.from_pretrained(args.tokenizer_id,subfolder="feature_extractor")
    pipeline = StableDiffusionPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        unet=model.unet,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        scheduler=model.diffusion_schedule,
        safety_checker=None,
    ).to(args.device)
    del ckpt
    del model
    return pipeline

def collect_fn(batchs):
    prompt = [batch["prompt"] for batch in batchs]
    filename = [batch["filename"] for batch in batchs]
    return {"prompt": prompt, "filename": filename}

def get_dataloader(root, mask_range, height, width, batch_size):
    dataset = EvalDataset(root=root, mask_range=mask_range, height=height, width=width)
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
    os.makedirs(args.output_dir,exist_ok=True)
    dataloader = get_dataloader(
        root=args.root,
        mask_range=args.mask_range,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
    )
    # seed_everything(args.seed)
    for batch in dataloader:
        prompts = batch['prompt']
        filenames = batch['filename']


        images = pipeline.__call__(
            prompt=prompts,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images
        for image,filename in zip(images,filenames):
            save_path = os.path.join(args.output_dir,filename).replace(".png",".jpg")
            image.save(save_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
