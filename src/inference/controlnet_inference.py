import warnings

warnings.filterwarnings("ignore")
import argparse
import torch
import shutil
import os
from src.models.controlnet_module import ControlLitModule
import hydra
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPFeatureExtractor
from omegaconf import OmegaConf
import matplotlib.animation as animation
from diffusers import StableDiffusionControlNetPipeline
import imageio


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
        default="logs/train/runs/controlnet/2024-12-18_13-02-29/checkpoints/epoch=002-val_loss=0.1412.ckpt",
    )
    parser.add_argument(
        "--model_config", type=str, default="configs/model/controlnet.yaml"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/controlnet/",
    )
    parser.add_argument("--tokenizer_id", type=str, default="checkpoints/stablev15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--guidance_scale", type=int, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--save_denoising",
        action="store_true",
        help="Whether to save the denoising results",
    )
    parser.add_argument("--tmp_dir", type=str, default="tmp")
    parser.add_argument("--duration", type=int, default=1)
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
        "checkpoints/stablev15", subfolder="feature_extractor"
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


def get_gif(args):
    filenames = [
        os.path.join(args.tmp_dir, f"{i}.png")
        for i in range(0, args.num_inference_steps)
    ]
    base_name = os.path.basename(args.mask).split(".")[0]
    save_path = os.path.join(args.output_dir, f"denoising_process_{base_name}.gif")
    with imageio.get_writer(save_path, mode="I", duration=args.duration) as writer:
        for filename in filenames:
            im = imageio.imread(filename)
            writer.append_data(im)
    print(f"{save_path} has saved!")
    shutil.rmtree(args.tmp_dir)
    print(f"temp dir has been removed!")


def main():
    args = get_args()
    print("init pipeline...")
    pipeline: StableDiffusionControlNetPipeline = get_pipeline(args)
    print("start inference...")

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
    generator = torch.Generator().manual_seed(args.seed)

    def decode_tensors(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        image = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor,
            return_dict=False,
            generator=generator,
        )[0]
        image, has_nsfw_concept = pipe.run_safety_checker(
            image, args.device, torch.float32
        )
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = pipe.image_processor.postprocess(
            image, output_type="pil", do_denormalize=do_denormalize
        )[0]

        tmp_save_path = os.path.join(args.tmp_dir, f"{step}.png")

        plt.imshow(image)
        plt.title(f"step:{step+1}")
        plt.axis("off")
        plt.savefig(tmp_save_path)
        plt.close()
        return callback_kwargs

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_denoising:
        print(f"create temp dir:{args.tmp_dir}")
        os.makedirs(args.tmp_dir, exist_ok=True)
        image = pipeline.__call__(
            prompt=args.prompt,
            image=mask,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            callback_on_step_end=decode_tensors,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
        get_gif(args)
    else:
        image = pipeline.__call__(
            prompt=args.prompt,
            image=mask,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]
    base_name = os.path.basename(args.mask)
    output_path = os.path.join(args.output_dir, base_name).replace(".png",".jpg")
    image.save(output_path)
    print(f"done.image saved to {output_path}")


if __name__ == "__main__":
    main()
