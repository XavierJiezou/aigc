from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from lightning import LightningDataModule
from src.data.components.mmcelebahq_xl_t2i import MMCelebAHQ
from transformers import CLIPTokenizer
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PretrainedConfig


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(
    prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True
):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def compute_embeddings(
    batch,
    proportion_empty_prompts,
    text_encoders,
    tokenizers,
    is_train=True,
    resolution=512,
    crops_coords_top_left_h=0,
    crops_coords_top_left_w=0,
):
    original_size = (resolution, resolution)
    target_size = (resolution, resolution)
    crops_coords_top_left = (crops_coords_top_left_h, crops_coords_top_left_w)
    prompt_batch = batch["prompts"]

    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
    )
    
    add_text_embeds = pooled_prompt_embeds

    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])

    add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)

    unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}


class MMCelebAHQDataModule(LightningDataModule):

    def __init__(
        self,
        root="data/mmcelebahq",
        face_file=None,
        mask_file=None,
        text_file="data/mmcelebahq/text.json",
        pretrained_model_name_or_path="checkpoints/stable-diffusion-xl-base-1.0",
        revision=None,
        variant=None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        proportion_empty_prompts=0.0,
    ) -> None:
        """Initialize a `MNISTDataModule`."""
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[MMCelebAHQ] = None
        self.val_dataset: Optional[MMCelebAHQ] = None
        self.test_dataset: Optional[MMCelebAHQ] = None

        self.batch_size_per_device = batch_size

        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path, revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path, revision, subfolder="text_encoder_2"
        )

        self.text_encoder_one = text_encoder_cls_one.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
            variant=variant,
        )
        self.text_encoder_two = text_encoder_cls_two.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=revision,
            variant=variant,
        )
        # Load the tokenizers
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision,
            use_fast=False,
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=revision,
            use_fast=False,
        )

        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)

        # self.tokenizer_one.requires_grad_(False)
        # self.tokenizer_two.requires_grad_(False)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        MMCelebAHQ(
            root=self.hparams.root,
            face_file=self.hparams.face_file,
            mask_file=self.hparams.mask_file,
            text_file=self.hparams.text_file,
            split="train",
        )

        MMCelebAHQ(
            root=self.hparams.root,
            face_file=self.hparams.face_file,
            mask_file=self.hparams.mask_file,
            text_file=self.hparams.text_file,
            split="val",
        )

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            self.train_dataset = MMCelebAHQ(
                root=self.hparams.root,
                face_file=self.hparams.face_file,
                mask_file=self.hparams.mask_file,
                text_file=self.hparams.text_file,
                split="train",
            )

            self.val_dataset = self.test_dataset = MMCelebAHQ(
                root=self.hparams.root,
                face_file=self.hparams.face_file,
                mask_file=self.hparams.mask_file,
                text_file=self.hparams.text_file,
                split="val",
            )

    def collate_fn(self, examples):
        pixel_values = [example["pixel_values"] for example in examples]
        conditioning_pixel_values = [
            example["conditioning_pixel_values"] for example in examples
        ]
        prompts = [example["prompt"] for example in examples]
        batch = {"prompts": prompts}
        prompt_embeds = compute_embeddings(
            batch=batch,
            proportion_empty_prompts=self.hparams.proportion_empty_prompts,
            text_encoders=[self.text_encoder_one, self.text_encoder_two],
            tokenizers=[self.tokenizer_one, self.tokenizer_two],
            is_train=True,
        )


        prompt_ids = torch.stack([prompt_embed for prompt_embed in prompt_embeds["prompt_embeds"]])
        add_text_embeds = torch.stack([text_embeds for text_embeds in prompt_embeds["text_embeds"]])
        add_time_ids = torch.stack([time_ids for time_ids in prompt_embeds["time_ids"]])

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        conditioning_pixel_values = torch.stack(conditioning_pixel_values)
        conditioning_pixel_values = conditioning_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        batch = {
            "conditioning_pixel_values": conditioning_pixel_values,
            "pixel_values": pixel_values,
            "prompt_ids":prompt_ids,
            "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
        }
        return batch

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
