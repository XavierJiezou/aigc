import argparse
import torch
import os
from src.models.diffusion_module import DiffusionLitModule
import hydra
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPTokenizer
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a photo of a corgi dog")
    parser.add_argument("--ckpt_path", type=str, default="tmp.ckpt")
    parser.add_argument(
        "--model_config", type=str, default="configs/model/stable_diffusion.yaml"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/stable_diffusion_dream_booth/corgi.png",
    )
    parser.add_argument(
        "--tokenizer_id", type=str, default="CompVis/stable-diffusion-v1-4"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--guidance_scale", type=int,default=7)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    args = parser.parse_args()
    return args


def get_pipeline(args):
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model_config = OmegaConf.load(args.model_config)  # 加载model config file
    model: DiffusionLitModule = hydra.utils.instantiate(model_config)
    model.load_state_dict(ckpt["state_dict"])
    tokenizer = CLIPTokenizer.from_pretrained(
        args.tokenizer_id,
        subfolder="tokenizer",
    )
    pipeline = StableDiffusionPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        unet=model.unet,
        feature_extractor=model.feature_extractor,
        tokenizer=tokenizer,
        scheduler=model.diffusion_schedule,
        safety_checker=None,
    )
    del ckpt
    del model
    return pipeline


def main():
    args = get_args()
    print("init pipeline...")
    pipeline:StableDiffusionPipeline = get_pipeline(args)
    print("start inference...")
    image = pipeline.__call__(args.prompt,height=args.height,width=args.width,num_inference_steps=args.num_inference_steps).images[0]
    os.makedirs(os.path.dirname(args.output_path),exist_ok=True)
    image.save(args.output_path)
    print(f"done.image saved to {args.output_path}")


if __name__ == "__main__":
    main()
