import torch
import torch.nn as nn

from transformers import CLIPImageProcessor
from .clip import CLIP

class CLIPVisionTower(nn.Module):
    def __init__(self, args, img_size=512, delay_load=False):
        super().__init__()

        # test
        if hasattr(args, 'mm_vision_tower'):
            self.clip_model = args.mm_vision_tower
        else: # train
            self.clip_model = args.vision_tower
        self.is_loaded = False
        self.img_size = img_size

        if not delay_load:
            self.load_model()

    def load_model(self):
        self.image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge":self.img_size}, resample=3,  do_center_crop=True, crop_size={"height": self.img_size, "width": self.img_size},
                                                  do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                  image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )

        self.vision_tower = CLIP()

        self.vision_tower.load_state_dict(torch.load(self.clip_model),strict=False)
        
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):

        if type(images) is list:
            image_features = []
            image_features_dict = []
            for image in images:
                image_feature_dict = self.vision_tower(image.unsqueeze(0))
                image_features_dict.append(image_feature_dict)
                image_feature = image_feature_dict['res4']
                image_feature = image_feature.reshape(*image_feature.shape[:2],-1).permute(0,2,1)
                image_features.append(image_feature)
        else:
            image_features_dict = self.vision_tower(images)
            image_features = image_features_dict['res4']
            image_features = image_features.reshape(*image_features.shape[:2],-1).permute(0,2,1)

        return image_features, image_features_dict

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device
