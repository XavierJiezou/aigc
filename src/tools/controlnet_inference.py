import argparse
import torch
import os
from src.models.controlnet_module import ControlLitModule
import hydra
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPFeatureExtractor
from omegaconf import OmegaConf
from diffusers import StableDiffusionControlNetPipeline

# torch.set_float32_matmul_precision('high')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="She wears heavy makeup. She has mouth slightly open. She is smiling, and attractive.",
    )
    parser.add_argument("--mask", default="data/mmcelebahq/mask/27504.png")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/train/runs/controlnet/2024-12-18_00-35-44/checkpoints/epoch=000-val_loss=0.0000.ckpt",
    )
    parser.add_argument(
        "--model_config", type=str, default="configs/model/controlnet.yaml"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/controlnet/face.png",
    )
    parser.add_argument("--tokenizer_id", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--guidance_scale", type=int, default=7)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    args = parser.parse_args()
    return args


def get_pipeline(args):
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model_config = OmegaConf.load(args.model_config)  # 加载model config file
    model: ControlLitModule = hydra.utils.instantiate(model_config)
    model.load_state_dict(ckpt["state_dict"])
    tokenizer = CLIPTokenizer.from_pretrained(
        args.tokenizer_id,
        subfolder="tokenizer",
    )
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "checkpoints", subfolder="feature_extractor"
    )
    pipeline = StableDiffusionControlNetPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        unet=model.unet,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        scheduler=model.diffusion_schedule,
        safety_checker=None,
        controlnet=model.controlnet,
    ).to(args.device)
    # pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    # pipeline.enable_xformers_memory_efficient_attention()
    del ckpt
    del model
    return pipeline


def main():
    args = get_args()
    print("init pipeline...")
    pipeline: StableDiffusionControlNetPipeline = get_pipeline(args)
    print("start inference...")
    generator = torch.Generator().manual_seed(args.seed)

    mask = Image.fromarray(np.array(Image.open(args.mask)), mode="L")

    mask_transforms = transforms.Compose(
        [
            transforms.Resize((args.height, args.width)),
            transforms.CenterCrop((args.height, args.width)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float()),  # 转换为 float 类型
        ]
    )
    mask = mask_transforms(mask)
    image = pipeline.__call__(
        prompt=args.prompt,
        image=mask,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    ).images[0]
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    image.save(args.output_path)
    print(f"done.image saved to {args.output_path}")


if __name__ == "__main__":
    main()
