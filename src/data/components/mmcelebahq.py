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

class MMCelebAHQ(Dataset):
    def __init__(
        self,
        dataset_path="data/mmcelebahq",
        split: Literal["train", "val"] = "train",
        face_file="data/mmcelebahq/face.zip",
        mask_file="data/mmcelebahq/mask.zip",
        text_file="data/mmcelebahq/text.json",
        size=512,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.face_file = None
        self.mask_file = None
        self.text_file = None

        self.filenames = self.get_filenames(split, face_file, mask_file, text_file)
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
                    filename: Image.fromarray(np.array(Image.open(filename)), mode="L")
                    for filename in mask_files
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
        filename = os.path.join(self.dataset_path, "face", f"{filename}.jpg")
        image = Image.open(filename).convert("RGB")
        return image

    def get_mask(self, filename):
        filename = os.path.join(self.dataset_path, "mask", f"{filename}.png")
        mask = Image.fromarray(np.array(Image.open(filename)), mode="L")
        return mask

    def get_text(self, filename):
        filename = os.path.join(self.dataset_path, "text", f"{filename}.txt")
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

        if self.face_file is not None:
            face_name = os.path.join(self.dataset_path, "face", f"{index}.jpg")
            image = self.face_file[face_name]
        else:
            image = self.get_image(index)

        if self.mask_file is not None:
            mask_name = os.path.join(self.dataset_path, "mask", f"{index}.png")
            mask = self.mask_file[mask_name]
        else:
            mask = self.get_mask(index)

        if self.text_file is not None:
            text_name = f"{index}.jpg"
            prompts: list = self.text_file[text_name]
            prompt = random.choices(prompts)[0]
        else:
            prompt = self.get_text(index)

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
    

def show_mmcelebahq():
    tokenizer = CLIPTokenizer.from_pretrained(
        "checkpoints",
        subfolder="tokenizer",
    )
    dataset = MMCelebAHQ(split="val")
    data = dataset[0]
    # print(
    #     data["instance_images"].shape,
    #     data["instance_masks"].shape,
    #     data["instance_prompt_ids"],
    # )
    image: torch.Tensor = data["instance_images"]
    mask: torch.Tensor = data["instance_masks"]
    prompt = data["instance_prompt_ids"]
    text = tokenizer.decode(prompt, skip_special_tokens=True)

    mask = mask.squeeze().detach().numpy().astype(np.uint8)

    palette = np.array(
        [
            (0, 0, 0),
            (204, 0, 0),
            (76, 153, 0),
            (204, 204, 0),
            (51, 51, 255),
            (204, 0, 204),
            (0, 255, 255),
            (51, 255, 255),
            (102, 51, 0),
            (255, 0, 0),
            (102, 204, 0),
            (255, 255, 0),
            (0, 0, 153),
            (0, 0, 204),
            (255, 51, 153),
            (0, 204, 204),
            (0, 51, 0),
            (255, 153, 51),
            (0, 204, 0),
        ],
        dtype=np.uint8,
    )
    color_mask = palette[mask]
    color_mask = Image.fromarray(color_mask)

    image = (image * 0.5) + 0.5
    image = image.permute(1, 2, 0).detach().numpy()
    image = image * 255.0
    image = image.astype(np.uint8)

    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.suptitle(text)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(color_mask)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("0.jpg", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    # show_mmcelebahq()
    
    # Length test
    assert len(MMCelebAHQ(split="train")) == 27000
    assert len(MMCelebAHQ(split="val")) == 3000
    
    # Shape test
    dataset = MMCelebAHQ(split="val")
    data = dataset[0] # 27000
    assert data["instance_images"].shape == (3, 512, 512)
    assert data["instance_masks"].shape == (1, 512, 512)