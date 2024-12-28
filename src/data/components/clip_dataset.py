from torch.utils.data import Dataset
from typing import Literal
import os
from PIL import Image
import numpy as np
import clip

class CLipCelebAHQ(Dataset):
    def __init__(
        self,
        root="data/mmcelebahq",
        split: Literal["train", "val"] = "train",
        clip_config: dict = None,
    ):

        self.root = root
        self.split = split
        self.filename_id = self.get_filename()
        _, self.process = clip.load(**clip_config)

    def get_filename(self):
        filename_id = []
        if self.split == "train":
            filename_id = [i for i in range(0, 27000)]
        elif self.split == "val":
            filename_id = [i for i in range(27000, 30000)]
        return filename_id

    def __len__(self):
        return len(self.filename_id)

    def __getitem__(self, index):
        filename_id = self.filename_id[index]
        face_path = os.path.join(self.root, "face", f"{filename_id}.jpg")
        mask_path = os.path.join(self.root, "mask", f"{filename_id}.png")

        face = Image.open(face_path)
        face = self.process(face)
        mask = np.array(Image.open(mask_path))

        return {"face": face, "mask": mask}


def show(face: np.ndarray, mask: np.ndarray):
    from matplotlib import pyplot as plt

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
    face = face.detach().cpu().permute(1,2,0).numpy()
    face = face * 0.5 + 0.5
    face = face * 255
    face = np.clip(face,0,255)
    face = face.astype(np.uint8)
    face = Image.fromarray(face)

    plt.subplot(1, 2, 1)
    plt.imshow(face)
    plt.axis("off")
    plt.title("Face")

    color_mask = palette[mask]
    color_mask = Image.fromarray(color_mask)

    plt.subplot(1, 2, 2)
    plt.imshow(color_mask)
    plt.axis("off")
    plt.title("Mask")
    plt.tight_layout()
    plt.savefig("clip_dataset_show.png")


if __name__ == "__main__":
    clip_config = dict(name="ViT-B/32", download_root="checkpoints/clip", device="cpu")
    train_dataset = CLipCelebAHQ(clip_config=clip_config)
    assert len(train_dataset) == 27000

    val_dataset = CLipCelebAHQ(split="val", clip_config=clip_config)
    assert len(val_dataset) == 3000

    data = val_dataset[0]
    face = data["face"]
    mask = data["mask"]
    assert face.shape == (3, 224, 224),f"face shape is {face.shape}"
    assert mask.shape == (512, 512)
    assert np.max(mask) < 19
    assert np.min(mask) > -1
    print(np.unique(mask))  # [ 0  1  2  4  5  6  7  9 11 12 13 17]
    show(face, mask)
