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
from src.models.ipadapter_module import IPAdapterLitModule


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
        ipadapter: IPAdapterLitModule,
        device: str,
    ):
        self.sd_pipeline = sd_pipeline
        self.ipadapter = ipadapter
        self.device = device
        self.clip_image_processor = CLIPImageProcessor()

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(
                images=pil_image, return_tensors="pt"
            ).pixel_values
            clip_image_embeds = self.ipadapter.image_encoder(
                clip_image.to(self.device)
            ).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device)
        image_prompt_embeds = self.ipadapter.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.ipadapter.image_proj_model(
            torch.zeros_like(clip_image_embeds)
        )
        return image_prompt_embeds, uncond_image_prompt_embeds

    def do_mask_text_attn(
        self, mask_features: torch.Tensor, text_features: torch.Tensor
    ):
        mask_text = self.ipadapter.mask_text_attn.forward(
            x=mask_features, context=text_features
        )  # [bs,4,768] -> [bs,4,768]
        text_mask = self.ipadapter.text_mask_attn.forward(
            x=text_features, context=mask_features
        )  # [bs,77,768] -> [bs,77,768]

        encoder_hidden_states = torch.cat([mask_text, text_mask], dim=1)  # [bs,81,768]
        return encoder_hidden_states

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
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
            prompt_embeds = self.do_mask_text_attn(
                mask_features=image_prompt_embeds, text_features=prompt_embeds_
            )
            # prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            # negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
            negative_prompt_embeds = self.do_mask_text_attn(
                mask_features=uncond_image_prompt_embeds,
                text_features=negative_prompt_embeds_,
            )

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
    parser.add_argument(
        "--prompt",
        type=str,
        default="She wears heavy makeup. She has mouth slightly open. She is smiling, and attractive.",
    )
    parser.add_argument("--mask", default="data/mmcelebahq/mask/27504.png")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/train/runs/ipadapter/2024-12-18_21-30-54/checkpoints/epoch=002-val_loss=0.1439.ckpt",
    )
    parser.add_argument(
        "--model_config", type=str, default="configs/model/ipadapter.yaml"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ipadapter/",
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
    model: IPAdapterLitModule = hydra.utils.instantiate(model_config)
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
    pipeline = StableDiffusionPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        unet=model.unet,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        scheduler=model.diffusion_schedule,
        safety_checker=None,
    ).to(args.device)
    # pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    # pipeline.enable_xformers_memory_efficient_attention()
    ipadapter_pipe = IPAdapterPipeline(
        sd_pipeline=pipeline, ipadapter=model, device=args.device
    )
    return ipadapter_pipe


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
    pipeline: IPAdapterPipeline = get_pipeline(args)
    print("start inference...")

    mask = Image.fromarray(np.array(Image.open(args.mask)), mode="L")
    generator = get_generator(args.seed, args.device)
    os.makedirs(args.output_dir, exist_ok=True)

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

    if args.save_denoising:
        print(f"create temp dir:{args.tmp_dir}")
        os.makedirs(args.tmp_dir, exist_ok=True)
        image = pipeline.generate(
            pil_image=mask,
            prompt=args.prompt,
            image=mask,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            callback_on_step_end=decode_tensors,
            callback_on_step_end_tensor_inputs=["latents"],
        )[0]
        get_gif(args)
    else:
        image = pipeline.generate(
            pil_image=mask,
            prompt=args.prompt,
            image=mask,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
        )[0]

    base_name = os.path.basename(args.mask)
    output_path = os.path.join(args.output_dir, base_name)
    image.save(output_path)
    print(f"done.image saved to {output_path}")


if __name__ == "__main__":
    main()