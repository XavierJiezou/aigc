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
from transformers import CLIPImageProcessor
import torch


class HicoT2IDataset(Dataset):
    def __init__(
        self,
        root="data/mmcelebahq",
        split: Literal["train", "val"] = "train",
        text_file=None,
        tokenizer_id="checkpoints/stablev15",
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            tokenizer_id,
            subfolder="tokenizer",
        )
        self.root = root
        self.split = split
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

        # self.clip_image_processor = CLIPImageProcessor()

        self.filenames = self.get_filenames(split, text_file)
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def get_filenames(self, split, text_file=None):
        filenames = None
        if split == "train":
            filenames = [i for i in range(0, 27000)]
        else:
            filenames = [i for i in range(27000, 30000)]

        if text_file is not None:
            with open(text_file, mode="r") as f:
                text_json: dict = json.load(f)
            tmp = {
                k: v for k, v in text_json.items() if int(k.split(".")[0]) in filenames
            }
            self.text_file = tmp
            # print("load text file done.")

        return filenames

    def __len__(self):
        return len(self.filenames)

    def get_image(self, filename):
        filename = os.path.join(self.root, "face", f"{filename}.jpg")
        image = Image.open(filename).convert("RGB")
        return image

    def get_mask(self, filename):
        filename = os.path.join(self.root, "mask", f"{filename}.png")
        mask = Image.open(filename)
        return mask

    def get_text(self, filename):
        filename = os.path.join(self.root, "text", f"{filename}.txt")
        with open(filename, "r") as f:
            prompts = f.readlines()
        caption = random.choices(prompts)[0]
        caption = caption.strip()
        return caption

    def __getitem__(self, index):
        index = self.filenames[index]
        prompt = None
        image = self.get_image(index)
        mask = self.get_mask(index)
        mask = np.array(mask)

        if self.text_file is not None:
            text_name = f"{index}.jpg"
            prompts: list = self.text_file[text_name]
            prompt = random.choices(prompts)[0]
        else:
            prompt = self.get_text(index)

        # 查看都有哪些类别
        cls = np.unique(mask)

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
            "instance_images": self.image_transforms(image),
            "cls_mask": cls_mask,
            # "cls": cls,
            "cls_text": cls_text,
            "instance_prompt_ids": self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids,
        }

def show():
    tokenizer = CLIPTokenizer.from_pretrained("checkpoints/stablev15", subfolder="tokenizer")
    dataset = HicoT2IDataset(text_file="data/mmcelebahq/text.json")
    data = dataset[0]

    print(f"instance_images:{data['instance_images'].shape}")
    print(f"cls_mask:{data['cls_mask'].shape}")
    print(f"cls_text:{data['cls_text'].shape}")
    print(f"instance_prompt_ids:{data['instance_prompt_ids'].shape}")

    # 19 * 512 * 512
    import os

    os.makedirs("tmp",exist_ok=True)
    index = 0
    for text, mask in zip(data["cls_text"], data["cls_mask"]):
        mask = mask * 255 # 0 255
        mask = Image.fromarray(mask)
        text = tokenizer.decode(text,skip_special_tokens=True)
        if text == "":
            text = f"空_{index}"
            index += 1
        mask.save(f"tmp/{text}.png")
        # exit(0)
    
    



if __name__ == "__main__":
    show()