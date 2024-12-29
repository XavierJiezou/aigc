import torch
from torch.nn import functional as F
from torch import nn as nn
from clip.model import VisionTransformer


class MaskEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings=19,
        embedding_dim=3,
        image_resolution=224,
        vision_patch_size=16,
        vision_width=768,
        vision_layers=12,
        embed_dim=512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        vision_heads = vision_width // 64
        self.image_resolution = image_resolution
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
        )

    def forward(self, mask: torch.Tensor):
        mask = self.embedding(mask)  # bs 512 512 -> bs 512 512 3
        mask = mask.permute(0, 3, 1, 2) # bs 512 512 3 -> bs 3 512 512
        mask = F.interpolate(
            mask,
            size=(self.image_resolution, self.image_resolution),
            mode="bilinear",
            align_corners=False,
        ) # bs 3 512 512 -> bs 3 224 224

        mask_feature = self.visual(mask)
        return mask_feature
    
    def init_visual(self,clip:nn.Module):
        visual_state_dict = {}
        clip.named_parameters
        for name,v in clip.named_parameters():
            new_k:str = name
            if new_k[:7] == "visual.":
                new_k = new_k[7:]
                visual_state_dict[new_k] = v
        self.visual.load_state_dict(visual_state_dict,strict=True)

if __name__ == "__main__":
    model = MaskEncoder()
    model.init_visual(checkpoint="checkpoints/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth")
