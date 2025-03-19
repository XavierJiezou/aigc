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
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from src.models.components.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor
from src.models.ipadapter_raw_module import IPAdapterRawLitModule


def get_generator(seed, device):
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None
    return generator

class IPAdapterPipeline:
    def __init__(
        self,
        sd_pipeline: StableDiffusionPipeline,
        ipadapter: IPAdapterRawLitModule,
        device: str,
        size=512,
    ):
        self.sd_pipeline:StableDiffusionPipeline = sd_pipeline
        self.ipadapter = ipadapter
        self.device = device

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            clip_image_embeds = self.ipadapter.mask_encoder(pil_image.to(self.device))
        else:
            clip_image_embeds = clip_image_embeds.to(self.device)
        image_prompt_embeds = self.ipadapter.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def set_scale(self, scale):
        for attn_processor in self.sd_pipeline.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

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
        num_inference_steps=50,
        **kwargs,
    ):
        
        self.set_scale(scale)
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
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            # prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            # negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

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

class EvalDataset(Dataset):
    def __init__(self, root: str, mask_range=None, height=512, width=512):
        super().__init__()
        mask_range = mask_range or (27000, 30000)
        self.mask_range = [i for i in range(mask_range[0], mask_range[1])]
        self.root = root
        self.prompts = self.get_prompts()
        self.masks = self.get_masks()
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.CenterCrop((height, width)),
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x.long()),  # 转换为 int64 类型
            ]
        )

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
        mask = Image.fromarray(np.array(Image.open(mask_path)), mode="L")
        mask = self.mask_transforms(mask)

        return {"prompt": prompt, "filename": filename, "mask": mask}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/mmcelebahq",
    )
    parser.add_argument("--ckpt_path", type=str, default="logs/train/runs/ipadapter_raw/2024-12-29_21-30-22/checkpoints/epoch=049-val_loss=0.1400.ckpt")
    parser.add_argument("--model_config", type=str, default="configs/model/ipadapter_raw.yaml")
    parser.add_argument("--output_dir", type=str, default="data/visulization/ipadapter_raw",)
    parser.add_argument("--tokenizer_id", type=str, default="checkpoints/stablev15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance_scale", type=int, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--mask_range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=[27000, 30000],
        help="Range of mask indices to process",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--scale",type=float,default=0.5)
    parser.add_argument("--save_denoising", action="store_true", help="Whether to save the denoising results")
    parser.add_argument("--tmp_dir", type=str, default="tmp")
    parser.add_argument("--duration", type=int, default=1)
    args = parser.parse_args()
    return args

def collect_fn(batchs):
    prompt = [batch["prompt"] for batch in batchs]
    filename = [batch["filename"] for batch in batchs]
    mask = [batch["mask"] for batch in batchs]
    mask = torch.stack(mask).squeeze(dim=1)
    return {"prompt": prompt, "filename": filename, "mask": mask}



def get_dataloader(root, mask_range, height, width, batch_size):
    dataset = EvalDataset(root=root, mask_range=mask_range, height=height, width=width)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collect_fn
    )
    return dataloader

def get_pipeline(args):
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model_config = OmegaConf.load(args.model_config)  # 加载model config file
    model: IPAdapterRawLitModule = hydra.utils.instantiate(model_config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_id, subfolder="tokenizer")
    feature_extractor = CLIPFeatureExtractor.from_pretrained("checkpoints/stablev15", subfolder="feature_extractor")
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
    ipadapter_pipe = IPAdapterPipeline(sd_pipeline=pipeline, ipadapter=model, device=args.device)
    return ipadapter_pipe

def main():
    args = get_args()
    print("init pipeline...")
    pipeline: IPAdapterPipeline = get_pipeline(args)
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
    for batch in dataloader:
        prompts = batch['prompt']
        filenames = batch['filename']
        masks = batch['mask']
        images = pipeline.generate(
            pil_image=masks,
            prompt=prompts,
            image=masks,
            scale=args.scale,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_samples=1
        )
        for image,filename in zip(images,filenames):
            save_path = os.path.join(args.output_dir,filename).replace(".png",".jpg")
            image.save(save_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
