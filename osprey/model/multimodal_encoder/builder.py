import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, delay_load=False):

    return CLIPVisionTower(args=vision_tower_cfg)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
