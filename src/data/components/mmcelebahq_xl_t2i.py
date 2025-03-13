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


class MMCelebAHQ(Dataset):
    def __init__(
        self,
        root="data/mmcelebahq",
        split: Literal["train", "val"] = "train",
        face_file="data/mmcelebahq/face.zip",
        mask_file="data/mmcelebahq/mask.zip",
        text_file="data/mmcelebahq/text.json",
    ):

        self.root = root
        self.split = split
        self.face_file = None
        self.mask_file = None
        self.text_file = None

        # self.clip_image_processor = CLIPImageProcessor()

        self.filenames = self.get_filenames(split, face_file, mask_file, text_file)
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def get_filenames(self, split, face_file=None, mask_file=None, text_file=None):
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

        if mask_file is not None:
            with zipfile.ZipFile(mask_file, "r") as zip_ref:
                mask_files = zip_ref.namelist()
                mask_files = [
                    filename
                    for filename in mask_files
                    if filename.endswith(".png")
                    and int(os.path.basename(filename).split(".")[0]) in filenames
                ]
                mask_files = {
                    filename: np.array(Image.open(filename)) for filename in mask_files
                }
                self.mask_file = mask_files
            # print("load mask file done.")

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
        mask = Image.open(filename).convert("RGB")
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
        example = {}
        image = None
        mask = None
        prompt = None

        if self.face_file is not None:
            face_name = os.path.join(self.root, "face", f"{index}.jpg")
            image = self.face_file[face_name]
        else:
            image = self.get_image(index)

        if self.mask_file is not None:
            mask_name = os.path.join(self.root, "mask", f"{index}.png")
            mask = self.mask_file[mask_name]
        else:
            mask = self.get_mask(index)

        if self.text_file is not None:
            text_name = f"{index}.jpg"
            prompts: list = self.text_file[text_name]
            prompt = random.choices(prompts)[0]
        else:
            prompt = self.get_text(index)

        example["pixel_values"] = self.image_transforms(image)
        example["conditioning_pixel_values"] = self.conditioning_image_transforms(mask)
        example["prompt"] = prompt
        return example
