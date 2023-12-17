import torch
import torch.nn.functional as F
import torch.nn as nn

from open_clip.model import _build_vision_tower


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = 'convnext_large'

        vision_cfg = {'timm_model_name': model_name, 'timm_model_pretrained': False, 'timm_pool': '', 'timm_proj': 'mlp', 'timm_drop': 0.0, 'timm_drop_path': 0.1, 'image_size': 320}
        self.visual = _build_vision_tower(embed_dim=768, vision_cfg=vision_cfg, quick_gelu=False)

        self.eval()
        self.freeze_everything()

    def freeze_everything(self):
        for param in self.visual.parameters():
            param.requires_grad = False

    def extract_features(self, x):
        out = {}
        x = x.to(self.visual.trunk.stem.state_dict()['1.bias'].dtype)
        x = self.visual.trunk.stem(x)
        out['stem'] = x.contiguous() 
        for i in range(4):
            x = self.visual.trunk.stages[i](x)
            out[f'res{i+2}'] = x.contiguous() 
        
        x = self.visual.trunk.norm_pre(x)
        out['clip_vis_dense'] = x.contiguous()
        return out

    def forward(self, x):
        self.eval()
        with torch.no_grad():
            return self.extract_features(x)
    