from src.pipeline.pipeline_hico_controlnet import StableDiffusionHicoControlNetPipeline
from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
    ControlNetModel,
)
from transformers import CLIPTokenizer
import argparse
import numpy as np
from PIL import Image
import torch
import os


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
        default="logs/train/runs/hico_controlnet/2025-03-17_11-31-04/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="checkpoints/stablev15",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/hico_controlnet/",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance_scale", type=int, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--remove_mask_id", default=-1, type=int)
    args = parser.parse_args()
    return args


def getpipeline(args):

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


def get_cls_prompt_mask(mask_path, remove_mask_id=-1, tokenizer_id=None):
    categories = [
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
        "necklace",  #
        "neck",
        "clothing",
    ]

    mask = np.array(Image.open(mask_path))
    cls_mask = []
    cls_text = []
    for i, category in enumerate(categories):
        if i == remove_mask_id:
            print(f"remove class {categories[remove_mask_id]}!")
            continue
        tmp_mask = np.zeros_like(mask)
        tmp_mask[mask == i] = 1
        cls_mask.append(tmp_mask)
        # cls_text.append(self.categories[i])
        # 检查 tmp_mask 是否全为 0
        if np.any(tmp_mask):
            cls_text.append(category)
        else:
            cls_text.append("")
    cls_mask = np.stack(cls_mask)[np.newaxis]  # 1 19 512 512
    tokenizer = CLIPTokenizer.from_pretrained(
        tokenizer_id,
        subfolder="tokenizer",
    )
    cls_text = tokenizer(
        cls_text,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    cls_text = cls_text.unsqueeze(0)
    return cls_mask, cls_text


if __name__ == "__main__":
    args = get_args()
    print("init pipeline")
    pipeline: StableDiffusionHicoControlNetPipeline = getpipeline(args)
    generator = torch.Generator().manual_seed(args.seed)
    cls_mask, cls_text = get_cls_prompt_mask(args.mask, args.remove_mask_id,tokenizer_id=args.base_model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    image = pipeline.__call__(
        prompt=args.prompt,
        cls_text=cls_text,
        fuse_type="avg",
        image=cls_mask,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )[0][0]

    base_name = os.path.basename(args.mask)
    if args.remove_mask_id > -1:
        base_name = (
            base_name.split(".")[0] + f"_remove_mask_id_{args.remove_mask_id}.jpg"
        )
    output_path = os.path.join(args.output_dir, base_name).replace(".png", ".jpg")
    image.save(output_path)
    print(f"done.image saved to {output_path}")
