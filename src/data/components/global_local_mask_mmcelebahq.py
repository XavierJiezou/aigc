from PIL import Image
import os
import numpy as np
from transformers import CLIPTokenizer
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import torch
import json


class GlobalLocalMaskMMCelebAHQ(Dataset):
    def __init__(
        self,
        root="data/mmcelebahq",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        phase="train",
    ):
        self.root = root
        self.phase = phase
        self.face_paths, self.mask_paths, self.prompts = self.get_face_mask_prompt()
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.to_tensor = T.ToTensor()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "checkpoints/stablev15",
            subfolder="tokenizer",
        )

    def get_face_mask_prompt(self):
        file_ids = [i for i in range(0, 27000)]
        if self.phase != "train":
            file_ids = [i for i in range(27000, 30000)]

        face_paths = [os.path.join(self.root, "face", f"{i}.jpg") for i in file_ids]
        mask_paths = [os.path.join(self.root, "mask", f"{i}.png") for i in file_ids]
        with open(os.path.join(self.root, "text.json"), mode="r") as f:
            prompts = json.load(f)
        return face_paths, mask_paths, prompts

    def __len__(self):
        return len(self.face_paths)

    def __getitem__(self, idx):

        image = Image.open(self.face_paths[idx]).convert("RGB")
        prompts = self.prompts[f"{idx}.jpg"]
        description = random.choices(prompts, k=1)[0].strip()

        mask = np.array(Image.open(self.mask_paths[idx]))
        mask_list = [self.to_tensor(Image.open(self.mask_paths[idx]).convert("RGB"))]
        for i in range(19):
            local_mask = np.zeros_like(mask)
            local_mask[mask == i] = 255

            drop_image = random.random() < self.drop_image_prob
            if drop_image:
                local_mask = np.zeros_like(mask)

            local_mask_rgb = Image.fromarray(local_mask).convert("RGB")
            local_mask_tensor = self.to_tensor(local_mask_rgb)
            mask_list.append(local_mask_tensor)
        condition_img = torch.stack(mask_list, dim=0)

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        # drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        
        instance_prompt_ids = self.tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return {
            "image": self.to_tensor(image),
            "condition": condition_img,
            "instance_prompt_ids": instance_prompt_ids,
        }
