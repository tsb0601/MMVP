import os
from .clip_encoder import CLIPVisionTower
from .dino_encoder import DINOVisionTower


def build_vision_tower(vision_tower_cfg, load_model = "clip", **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        #return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        if load_model == "clip":
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)        
        else:
            return DINOVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)



    raise ValueError(f'Unknown vision tower: {vision_tower}')
