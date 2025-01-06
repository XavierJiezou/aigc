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


class IPAdapterDataset(Dataset):
    def __init__(
        self,
        root="data/mmcelebahq",
        split: Literal["train", "val"] = "train",
        face_file="data/mmcelebahq/face.zip",
        text_file="data/mmcelebahq/text.json",
        size=512,
        t_drop_rate=0.05,
        i_drop_rate=0.05,
        ti_drop_rate=0.05,
    ):

        self.root = root
        self.split = split
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.face_file = None
        self.text_file = None

        self.clip_image_processor = CLIPImageProcessor()

        self.filenames = self.get_filenames(split, face_file, text_file)
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        self.tokenizer = CLIPTokenizer.from_pretrained("checkpoints/stablev15", subfolder="tokenizer")
    

    def get_filenames(self, split, face_file=None,text_file=None):
        filenames = None
        if split == "train":
            filenames = [i for i in range(0, 27000)]
        else:
            filenames = [i for i in range(27000, 30000)]

        if face_file is not None:
            with zipfile.ZipFile(face_file, "r") as zip_ref:
                face_files = zip_ref.namelist()
                face_files = [
                    filename
                    for filename in face_files
                    if filename.endswith(".jpg")
                    and int(os.path.basename(filename).split(".")[0]) in filenames
                ]
                face_files = {filename: Image.open(filename) for filename in face_files}
                self.face_file = face_files
            # print("load face file done.")

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

    def get_text(self, filename):
        filename = os.path.join(self.root, "text", f"{filename}.txt")
        with open(filename, "r") as f:
            prompts = f.readlines()
        caption = random.choices(prompts)[0]
        caption = caption.strip()
        return caption

    def __getitem__(self, index):
        index = index if self.split == "train" else index + 27000
        raw_image = None
        prompt = None

        if self.face_file is not None:
            face_name = os.path.join(self.root, "face", f"{index}.jpg")
            raw_image = self.face_file[face_name]
        else:
            raw_image = self.get_image(index)
        
        image = self.transform(raw_image)
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        if self.text_file is not None:
            text_name = f"{index}.jpg"
            prompts: list = self.text_file[text_name]
            prompt = random.choices(prompts)[0]
        else:
            prompt = self.get_text(index)

        # random drop reference to:https://github.com/tencent-ailab/IP-Adapter/blob/main/tutorial_train.py#L102
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            prompt = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            prompt = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed
        }



if __name__ == "__main__":
    pass

