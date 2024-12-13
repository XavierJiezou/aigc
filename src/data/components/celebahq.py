from torch.utils.data import Dataset
from transformers import CLIPTokenizer
from typing import Literal
import os
from PIL import Image
import numpy as np
import random
from torchvision import transforms


class CelebahqDataset(Dataset):
    def __init__(
        self,
        dataset_path="data/mmcelebahq",
        stage: Literal["train", "val"] = "train",
    ):
        self.dataset_path = dataset_path
        self.dataset = self.get_dataset(dataset_path, stage)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="tokenizer",
        )
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def get_dataset(self, dataset_path, stage):
        file_path = os.path.join(dataset_path, stage + ".txt")
        with open(file_path, "r") as f:
            filenames = [line.strip() for line in f.readlines()]
        return filenames

    def __len__(self):
        return len(self.dataset)

    def get_image(self, filename):
        filename = os.path.join(self.dataset_path, "image", f"{filename}.jpg")
        image = np.array(Image.open(filename))
        return image

    def get_mask(self, filename):
        filename = os.path.join(self.dataset_path, "mask", f"{filename}.png")
        mask = np.array(Image.open(filename)).astype(np.float32)
        return mask

    def get_caption(self, filename):
        filename = os.path.join(self.dataset_path, "caption", f"{filename}.txt")
        with open(filename, "r") as f:
            prompts = f.readlines()
        caption = random.choices(prompts)[0]
        caption = caption.strip()
        return caption

    def __getitem__(self, index):

        example = {}
        image = self.get_image(index)
        mask = self.get_mask(index)
        prompt = self.get_caption(index)
        example["instance_images"] = self.transforms(image)
        example["instance_masks"] = mask
        example["instance_prompt_ids"] = self.tokenizer(
            prompt,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example


if __name__ == "__main__":

    tokenizer = CLIPTokenizer.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="tokenizer",
    )
    dataset = CelebahqDataset()
    from matplotlib import pyplot as plt
    plt.figure(figsize=(12, 8))
    for data in dataset:

        instance_prompt_ids = data['instance_prompt_ids']
        original_text = tokenizer.decode(instance_prompt_ids[0], skip_special_tokens=True)
        print(original_text)
        image = data["instance_images"]
        image = image * 0.5 + 0.5
        image = image * 255
        image = image.permute(1,2,0).numpy().astype(np.uint8)
        mask = data["instance_masks"]

        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title(original_text)
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(mask)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"show.png")
        break
