from torch.utils.data import Dataset
from transformers import CLIPTokenizer
from typing import Literal
import os
from PIL import Image
import numpy as np
import zipfile
import json
import random
from torchvision import transforms
import torch


class MMCelebahq(Dataset):
    def __init__(
        self,
        dataset_path="data/mmcelebahq",
        split: Literal["train", "val"] = "train",
        face_cache="data/mmcelebahq/face.zip",
        mask_cache="data/mmcelebahq/mask.zip",
        text_cache="data/mmcelebahq/text.json",
        size=512,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.face_cache = None
        self.mask_cache = None
        self.text_cache = None

        self.filenames = self.get_filenames(split, face_cache, mask_cache, text_cache)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "checkpoints",
            subfolder="tokenizer",
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x.float()),  # 转换为 float 类型
            ]
        )
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def get_filenames(self, split, face_cache=None, mask_cache=None, text_cache=None):
        filenames = None
        if split == "train":
            filenames = [i for i in range(0, 27000)]
        else:
            filenames = [i for i in range(27000, 30000)]

        if face_cache is not None:
            with zipfile.ZipFile(face_cache, "r") as zip_ref:
                face_files = zip_ref.namelist()
                face_files = [
                    filename
                    for filename in face_files
                    if filename.endswith(".jpg") and int(os.path.basename(filename).split(".")[0]) in filenames
                ]
                face_files = {filename: Image.open(filename) for filename in face_files}
                self.face_cache = face_files
            print("load face cache done.")

        if mask_cache is not None:
            with zipfile.ZipFile(mask_cache, "r") as zip_ref:
                mask_files = zip_ref.namelist()
                mask_files = [
                    filename
                    for filename in mask_files
                    if filename.endswith(".png") and int(os.path.basename(filename).split(".")[0]) in filenames
                ]
                mask_files = {
                    filename: Image.fromarray(np.array(Image.open(filename)), mode="L")
                    for filename in mask_files
                }
                self.mask_cache = mask_files
            print("load mask cache done.")

        if text_cache is not None:
            with open(text_cache, mode="r") as f:
                text_json: dict = json.load(f)
            tmp = {k: v for k, v in text_json.items() if int(k.split(".")[0]) in filenames}
            self.text_cache = tmp
            print("load text cache done.")

        return filenames

    def __len__(self):
        return len(self.filenames)

    def get_image(self, filename):
        filename = os.path.join(self.dataset_path, "image", f"{filename}.jpg")
        image = Image.open(filename)
        return image

    def get_mask(self, filename):
        filename = os.path.join(self.dataset_path, "mask", f"{filename}.png")
        mask = Image.fromarray(np.array(Image.open(filename)), mode="L")
        return mask

    def get_caption(self, filename):
        filename = os.path.join(self.dataset_path, "caption", f"{filename}.txt")
        with open(filename, "r") as f:
            prompts = f.readlines()
        caption = random.choices(prompts)[0]
        caption = caption.strip()
        return caption

    def __getitem__(self, index):
        index = index if self.split == "train" else index + 27000
        example = {}
        image = None
        mask = None
        prompt = None

        if self.face_cache is not None:
            face_name = os.path.join(self.dataset_path,"face",f"{index}.jpg")
            image = self.face_cache[face_name]
        else:
            image = self.get_image(index)

        if self.mask_cache is not None:
            mask_name = os.path.join(self.dataset_path,"mask",f"{index}.png")
            mask = self.mask_cache[mask_name]
        else:
            mask = self.get_mask(index)
        
        if self.text_cache is not None:
            text_name = f"{index}.jpg"
            prompts:list = self.text_cache[text_name]
            prompt = random.choices(prompts)[0]
        else:
            prompt = self.get_caption(index)

        mask = self.mask_transforms(mask)
        example["instance_images"] = self.transforms(image)
        example["instance_masks"] = mask
        example["instance_prompt_ids"] = self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example


if __name__ == "__main__":

    tokenizer = CLIPTokenizer.from_pretrained(
        "checkpoints",
        subfolder="tokenizer",
    )
    dataset = MMCelebahq(split="val")
    data = dataset[0]
    print(data['instance_images'].shape,data['instance_masks'].shape,data['instance_prompt_ids'])
    image:torch.Tensor = data['instance_images']
    mask:torch.Tensor = data['instance_masks']
    prompt = data['instance_prompt_ids']
    text = tokenizer.decode(prompt)
    
    mask = mask.squeeze().detach().numpy().astype(np.uint8)
    mask = Image.fromarray(mask).convert("L")

    image = (image * 0.5) + 0.5
    image = image.permute(1,2,0).detach().numpy()
    image = image * 255.
    image = image.astype(np.uint8)

    from matplotlib import pyplot as plt

    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title(text,fontsize=8)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("0.png")



