import gc

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

models = {
  'vit_b': './checkpoints/sam_vit_b_01ec64.pth',
  'vit_l': './checkpoints/sam_vit_l_0b3195.pth',
  'vit_h': './checkpoints/sam_vit_h_4b8939.pth'
}

def get_sam_predictor(model_type='vit_b', device='cuda'):
  # sam model
  sam = sam_model_registry[model_type](checkpoint=models[model_type])
  sam = sam.to(device)

  predictor = SamPredictor(sam)

  return predictor

def get_mask_generator(model_type='vit_b', device='cuda'):
  sam = sam_model_registry[model_type](checkpoint=models[model_type])
  sam = sam.to(device)
  mask_generator = SamAutomaticMaskGenerator(
      model=sam)
  return mask_generator

def run_inference(predictor: SamPredictor, input_x, selected_points,
                  multi_object: bool = False):

  if len(selected_points) == 0:
    return []

  predictor.set_image(input_x)

  points = torch.Tensor(
      [p for p, _ in selected_points]
  ).to(predictor.device).unsqueeze(0)

  labels = torch.Tensor(
      [int(l) for _, l in selected_points]
  ).to(predictor.device).unsqueeze(0)

  transformed_points = predictor.transform.apply_coords_torch(
      points, input_x.shape[:2])
  # print(transformed_points.shape)
  # predict segmentation according to the boxes
  masks, scores, logits = predictor.predict_torch(
    point_coords=transformed_points,
    point_labels=labels,
    multimask_output=False,
  )
  masks = masks.cpu().detach().numpy()

  gc.collect()
  torch.cuda.empty_cache()

  return masks


def predict_box(predictor: SamPredictor, input_x, input_box):
  predictor.set_image(input_x)

  input_boxes = torch.tensor(input_box[None, :], device=predictor.device)
  transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, input_x.shape[:2])

  masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes = transformed_boxes,
    multimask_output = False
  )
  masks = masks.cpu().detach().numpy()
  
  gc.collect()
  torch.cuda.empty_cache()
  return masks