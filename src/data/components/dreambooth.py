from torch.utils.data import Dataset
from transformers import CLIPTokenizer
from datasets import load_from_disk
from torchvision import transforms


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        dataset_path="data/lewtun/corgi",
        size=512,
        instance_prompt="a photo of ccorgi dog",
        tokenizer=None,
    ):
        self.dataset = load_from_disk(dataset_path)
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.size = size
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        image = self.dataset[index]["image"]
        example["instance_images"] = self.transforms(image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example


if __name__ == "__main__":
    
    tokenizer = CLIPTokenizer.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="tokenizer",
    )
    dataset = DreamBoothDataset(tokenizer=tokenizer)
    print(dataset[0])
    print(f"Decoded tokens: {tokenizer.decode(dataset[0]['instance_prompt_ids'])}")

    print(f"Start of text token id: {tokenizer.cls_token_id}")
    print(f"End of text token id: {tokenizer.sep_token_id}")

