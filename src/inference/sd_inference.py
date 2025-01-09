import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import os
import imageio
import shutil
from src.models.diffusion_module import DiffusionLitModule
import hydra
from matplotlib import pyplot as plt
from transformers import CLIPTokenizer
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="She is wearing lipstick. She is attractive and has straight hair.")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument(
        "--model_config", type=str, default="configs/model/stable_diffusion.yaml"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/stable_diffusion",
    )
    parser.add_argument("--image_name",type=str,default="27000.png")
    parser.add_argument(
        "--tokenizer_id", type=str, default="checkpoints/stablev15"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
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


def get_gif(args):
    filenames = [
        os.path.join(args.tmp_dir, f"{i}.png")
        for i in range(0, args.num_inference_steps)
    ]
    save_path = os.path.join(args.output_dir, f"denoising_process_{args.image_name}.gif")
    with imageio.get_writer(save_path, mode="I", duration=args.duration) as writer:
        for filename in filenames:
            im = imageio.imread(filename)
            writer.append_data(im)
    print(f"{save_path} has saved!")
    shutil.rmtree(args.tmp_dir)
    print(f"temp dir has been removed!")


def get_pipeline(args):
    ckpt = None
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model_config = OmegaConf.load(args.model_config)  # 加载model config file
    model: DiffusionLitModule = hydra.utils.instantiate(model_config)
    if ckpt is not None:
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
    ).to(args.device)
    del ckpt
    del model
    return pipeline


def main():
    args = get_args()
    print("init pipeline...")
    pipeline: StableDiffusionPipeline = get_pipeline(args)
    print("start inference...")
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
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]
    output_path = os.path.join(args.output_dir, args.image_name)
    image.save(output_path)
    print(f"done.image saved to {output_path}")


if __name__ == "__main__":
    main()
