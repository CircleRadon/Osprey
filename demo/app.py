import os
from typing import List

import cv2
import gradio as gr
import numpy as np
import torch
from inference import SamPredictor, get_sam_predictor, run_inference, get_mask_generator, predict_box
from osprey_inference import Osprey
from matplotlib import pyplot as plt
from PIL import Image
import time
import argparse

parser = argparse.ArgumentParser(description='osprey demo', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model', help='path to osprey model', default='./checkpoints/osprey_v1.5_7b')
args = parser.parse_args()

osprey_ = Osprey(model_path=args.model)
sam_predictor = get_sam_predictor()
mask_generator = get_mask_generator()

def gen_colored_masks(
    annotation,
    random_color=False,
):
    """
    Code is largely based on https://github.com/CASIA-IVA-Lab/FastSAM/blob/4d153e909f0ad9c8ecd7632566e5a24e21cf0071/utils/tools_gradio.py#L130
    """
    device = annotation.device
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]

    index = (annotation != 0).to(torch.long).argmax(dim=0)
    if random_color:
        color = torch.rand((mask_sum, 1, 1, 3)).to(device)
    else:
        color = torch.ones((mask_sum, 1, 1, 3)).to(device) * torch.tensor(
            [30 / 255, 144 / 255, 255 / 255]
        ).to(device)
    transparency = torch.ones((mask_sum, 1, 1, 1)).to(device) * 0.6
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual

    mask = torch.zeros((height, weight, 4)).to(device)
    h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))

    mask[h_indices, w_indices, :] = mask_image[indices]
    mask_cpu = mask.cpu().numpy()

    return mask_cpu, sorted_indices


class ImageSketcher(gr.Image):
    """
    Code is from https://github.com/jshilong/GPT4RoI/blob/7c157b5f33914f21cfbc804fb301d3ce06324193/gpt4roi/app.py#L365

    Fix the bug of gradio.Image that cannot upload with tool == 'sketch'.
    """

    is_template = True  # Magic to make this work with gradio.Block, don't remove unless you know what you're doing.

    def __init__(self, **kwargs):
        super().__init__(tool='boxes', **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == 'boxes' and self.source in ['upload', 'webcam']:
            if isinstance(x, str):
                x = {'image': x, 'boxes': []}
            else:
                assert isinstance(x, dict)
                assert isinstance(x['image'], str)
                assert isinstance(x['boxes'], list)
        x = super().preprocess(x)
        return x

def mask_foreground(mask, trans=0.6, random_color=True):
  if random_color:
    color = np.concatenate([np.random.random(3)*255,np.array([trans*255])], axis=0)
  else:
    color = np.array([30, 144, 255, trans*255])
  h,w = mask.shape[-2:]
  mask_image = mask.reshape(h,w,1)*color.reshape(1,1,-1)

  return mask_image

def mask_background(mask, trans=0.5):
  h,w = mask.shape[-2:]
  mask_image = (1 - mask.reshape(h,w,1)) * np.array([0,0,0,trans*255])

  return mask_image

def img_add_masks(img_, colored_mask, mask, linewidth=2):
  if type(img_) is np.ndarray:
    img = Image.fromarray(img_, mode='RGB').convert('RGBA')
  else:
    img = img_.copy()
  h, w = img.height, img.width
  # contour
  temp = np.zeros((h, w, 1))
  contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(temp, contours, -1, (255, 255, 255), linewidth)
  color = np.array([1, 1, 1, 1])
  contour_mask = temp * color.reshape(1, 1, -1)
  
  overlay_inner = Image.fromarray(colored_mask.astype(np.uint8), 'RGBA')
  img.paste(overlay_inner, (0, 0), overlay_inner)

  overlay_contour = Image.fromarray(contour_mask.astype(np.uint8), 'RGBA')
  img.paste(overlay_contour, (0, 0), overlay_contour)
  return img

def img_select_point(original_img: np.ndarray,
                 sel_pix: list,
                 evt: gr.SelectData):
  img = original_img.copy()
  sel_pix.clear()
  sel_pix.append((evt.index, 1))   # append the foreground_point

  masks = run_inference(sam_predictor, original_img, sel_pix)
  # draw points
  for point, label in sel_pix:
    cv2.circle(img, point, 5, (240,240,240), -1, 0)
    cv2.circle(img, point, 5, (30,144,255), 2, 0)

  colored_mask = mask_foreground(masks[0][0])

  res = img_add_masks(original_img, colored_mask, masks[0][0])

  return img, res, masks[0][0]

def sam_everything(original_img):
  if original_img is None:
    raise gr.Error("Please upload an image first!")
  image = Image.fromarray(original_img, mode='RGB')
  h,w = original_img.shape[:2]
  masks = mask_generator.generate(original_img)
  mask_list = []
  for i, mask in enumerate(masks):
    mask_list.append(mask['segmentation'])

  mask_np = np.array(mask_list)

  mask_torch = torch.from_numpy(mask_np)
  inner_mask, order = gen_colored_masks(
      mask_torch,
      random_color=True,
  )

  contour_all = []
  temp = np.zeros((h, w, 1))
  for i, mask in enumerate(mask_np):
      annotation = mask.astype(np.uint8)
      contours, _ = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      for contour in contours:
          contour_all.append(contour)
  cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 1)
  color = np.array([0 / 255, 0 / 255, 0 / 255, 1])
  contour_mask = temp / 255 * color.reshape(1, 1, -1)

  image = image.convert('RGBA')
  overlay_inner = Image.fromarray((inner_mask * 255).astype(np.uint8), 'RGBA')
  image.paste(overlay_inner, (0, 0), overlay_inner)

  overlay_contour = Image.fromarray((contour_mask * 255).astype(np.uint8), 'RGBA')
  image.paste(overlay_contour, (0, 0), overlay_contour)

  
  return image, mask_list, image.copy(), order, ''

def init_image(img):
  if isinstance(img, dict):
    img = img['image']
  if isinstance(img, List):
    img = cv2.imread(img[0])
    img = img[:,:,::-1]
  
  h_, w_ = img.shape[:2]
  if h_>512:
    ratio = 512/h_
    new_h, new_w = int(h_*ratio), int(w_*ratio)
    preprocessed_img = cv2.resize(img, (new_w,new_h), interpolation=cv2.INTER_LINEAR)
  else:
    preprocessed_img = img.copy()
  
  return (preprocessed_img, preprocessed_img, preprocessed_img, preprocessed_img, [], None)


def mask_select_point(all_masks, output_mask_2_raw, mask_order, evt: gr.SelectData):
  h, w = output_mask_2_raw.height, output_mask_2_raw.width
  pointed_mask = None
  for i in range(len(mask_order)):
    idx = mask_order[i]
    msk = all_masks[idx]
    if msk[evt.index[1], evt.index[0]] == 1:
      pointed_mask = msk.copy()
      break
  
  if pointed_mask is not None:
    contours, hierarchy = cv2.findContours(pointed_mask.astype("uint8"),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    ret = output_mask_2_raw.copy()

    temp = np.zeros((h, w, 1))
    contours, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(temp, contours, -1, (255, 255, 255), 3)
    color = np.array([1, 1, 1, 1])
    contour_mask = temp * color.reshape(1, 1, -1)

    colored_mask = mask_background(pointed_mask)

    overlay_inner = Image.fromarray(colored_mask.astype(np.uint8), 'RGBA')
    ret.paste(overlay_inner, (0, 0), overlay_inner)

    overlay_contour = Image.fromarray(contour_mask.astype(np.uint8), 'RGBA')
    ret.paste(overlay_contour, (0, 0), overlay_contour)
  
    
    return ret, pointed_mask
  else:
    return output_mask_2_raw, None
  
def osprey_predict(img, pointed_mask, select):
  if pointed_mask is not None:
    caption = osprey_.osprey_predict(img, pointed_mask, select)
    return caption
  else:
    return "⚠️ No mask selected."
  
def gen_box_seg(inp):
  if inp is None:
    raise gr.Error("Please upload an image first!")
  image = inp['image']
  if len(inp['boxes'])==0:
    raise gr.Error("Please clear the raw boxes and draw a box first!")
  boxes = inp['boxes'][-1]
  

  input_box = np.array([boxes[0], boxes[1], boxes[2], boxes[3]]).astype(int)

  masks = predict_box(sam_predictor, image, input_box)

  colored_mask = mask_foreground(masks[0][0])

  res = img_add_masks(image, colored_mask, masks[0][0])
  
  return res, masks[0][0]


def radio_select(select):
  if select=='short description':
    return 0
  elif select == 'detailed description':
    return 1
  
def clear_image(img):
  print('clear')
  img = None
  return img

with gr.Blocks(title="Osprey") as demo:
  preprocessed_img = gr.State(value=None)

  # point
  selected_points = gr.State(value=[])
  point_mask = gr.State(value=None)

  box_mask = gr.State(value=None)

  # all
  all_masks = gr.State(value=None)
  output_mask_2_raw = gr.State(value=None)
  mask_order = gr.State(value=None)
  pointed_mask = gr.State(value=None)

  example_list = [
    [os.path.join(os.path.dirname(__file__), 'examples/house.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/mcdonald.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/vegetables.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/dinosaur.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/ice.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/wash.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/starbucks.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/olympic.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/baseball.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/roller_coaster.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/fountain.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/mirror.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/santa.jpg')],
    [os.path.join(os.path.dirname(__file__), 'examples/costume.jpg')]
  ]

  descrip_point = """
                    ### 💡 Tips:

                    🧸 Upload an image, and you can click on the image to select the area of interest.
              
                    🔖 In the bottom left, you can choose description with different levels of detail. Default is short description. 
                    
                    ⌛️ It takes about 1~ seconds to generate the segmentation result and the short description. The detailed description my take a longer time to 2~ seconds. The concurrency_count of queue is 1, please wait for a moment when it is crowded.
                    
                    🔔 If you want to choose another area, just click another point on the image.

                    📌 Click the button ❎ to clear the current image.
                    
                  """
  
  descrip_box = """
                    ### 💡 Tips:

                    🧸 Upload an image, and you can pull a frame on the image to select the area of interest.

                    🖱️ Then click the **Generate segmentation and description** button to generate segmentation and description accordingly.

                    🔔 If you want to choose another area or switch to another photo, click the button ↪️ first.

                    ❗️ If there are more than one box, the last one will be chosen.

                    🔖 In the bottom left, you can choose description with different levels of detail. Default is short description. 
                    
                    ⌛️ It takes about 1~ seconds to generate the segmentation result and the short description. The detailed description my take a longer time to 2~ seconds. The concurrency_count of queue is 1, please wait for a moment when it is crowded.

                    📌 Click the button **Clear Image** to clear the current image.
                    
                  """

  descrip_all = """
                    ### 💡 Tips:
                    🧸 Upload an image, and click the **Segment Everything** button to generate segmentation results.

                    🖱️ Then you can click on the masked image(👉right) to generate description accordingly.
                    
                    🔖 In the bottom left, you can choose description with different levels of detail. Default is short description. 
                    
                    ⌛️ It takes about 1.5~ seconds to generate the segmentation result and 0.8~ secondes of the short description. The concurrency_count of queue is 1, please wait for a moment when it is crowded.
                                        
                    📌 Click the button ❎ to clear the current image.
                  """

  # title
  with gr.Row():
    gr.HTML("""
            <h1 style="text-align: left; font-weight: 800; font-size: 2rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
            Osprey Demo
            <h2 style="text-align: left; font-weight: 600; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
            In this demo, we combine our model(Osprey) with <a href="https://segment-anything.com/" style="color:blue;">SAM</a>-ViT-B.
            """)
    
  with gr.TabItem("point-prompt"):
    # Segment image
    with gr.Row():
      with gr.Column():
        # input image
        input_image_1 = gr.Image(type="numpy", label='Input image', height=300)
        # point prompt
        radio_1 = gr.Radio(label='type', choices=['short description', 'detailed description'], value='short description')
        example_data1 = gr.Dataset(label='Examples', components=[input_image_1], samples=example_list)

      # show only mask
      with gr.Column():
        output_mask_1 = gr.Image(label='Segmentation', height=300)
        output_text_1 = gr.Textbox(
          label = 'Description(Osprey)'
        )
        gr.Markdown(descrip_point)

  with gr.TabItem("box-prompt"):
    with gr.Row():
      with gr.Column():
        input_image_3 = ImageSketcher(type="numpy", label='Input image', height=300)
        box_seg_button = gr.Button('Generate segmentation and description', variant='primary')
        clear_button = gr.Button('🗑 Clear Image')
        radio_3 = gr.Radio(label='type', choices=['short description', 'detailed description'], value='short description')
        example_data3 = gr.Dataset(label='Examples', components=[input_image_3], samples=example_list)

      with gr.Column():
        output_mask_3 = gr.Image(label='Segmentation', height=300)
        output_text_3 = gr.Textbox(
          label = 'Description(Osprey)'
        )

        gr.Markdown(descrip_box)
       
  with gr.TabItem("segment everything"):
    with gr.Row():
      with gr.Column():
        input_image_2 = gr.Image(type="numpy", label='Input image', height=300)

        segment_all = gr.Button('Segment Everything', variant='primary')
        radio_2 = gr.Radio(label='type', choices=['short description', 'detailed description'], value='short description')
        example_data2 = gr.Dataset(label='Examples', components=[input_image_2], samples=example_list)

      with gr.Column():
        output_mask_2 = gr.Image(label='Segmentation', height=300)
        output_text_2 = gr.Textbox(
          label = 'Description(Osprey)'
        )
        gr.Markdown(descrip_all)

  clear_button.click(lambda: None, [], [input_image_3]).then(
                lambda: None, None, None,
                _js='() => {document.body.innerHTML=\'<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>\'; setTimeout(function(){location.reload()},2000); return []}')
        
  input_image_1.upload(
      init_image,
      [input_image_1],
      [preprocessed_img, input_image_1, input_image_2, input_image_3, selected_points]
  )

  example_data1.click(
    init_image,
    [example_data1],
    [preprocessed_img, input_image_1, input_image_2, input_image_3, selected_points]
  )


  radio_1.select(
    radio_select,
    [radio_1],
    []
  )

  input_image_2.upload(
      init_image,
      [input_image_2],
      [preprocessed_img, input_image_1, input_image_2, input_image_3, selected_points]
  )

  example_data2.click(
    init_image,
      [example_data2],
      [preprocessed_img, input_image_1, input_image_2, input_image_3, selected_points]
  )

  radio_2.select(
    radio_select,
    [radio_2],
    []
  )

  input_image_3.upload(
      init_image,
      [input_image_3],
      [preprocessed_img, input_image_1, input_image_2, input_image_3, selected_points]
  )

  example_data3.click(
    init_image,
      [example_data3],
      [preprocessed_img, input_image_1, input_image_2, input_image_3, selected_points]
  )

  radio_3.select(
    radio_select,
    [radio_3],
    []
  )

  segment_all.click(
    sam_everything,
    [preprocessed_img],
    [output_mask_2, all_masks, output_mask_2_raw, mask_order, output_text_2]
  )

  box_seg_button.click(
    gen_box_seg,
    [input_image_3],
    [output_mask_3, box_mask]
  ).then(
    osprey_predict,
    [preprocessed_img, box_mask, radio_3],
    [output_text_3]
  )

  input_image_1.select(
      img_select_point,
      [preprocessed_img, selected_points],
      [input_image_1, output_mask_1, point_mask]
    ).then(
        osprey_predict,
        [preprocessed_img, point_mask, radio_1],
        [output_text_1]
    )
  
  output_mask_2.select(
    mask_select_point,
    [all_masks, output_mask_2_raw, mask_order],
    [output_mask_2, pointed_mask]
  ).then(
    osprey_predict,
    [preprocessed_img, pointed_mask, radio_2],
    [output_text_2]
  )

demo.queue(concurrency_count=1).launch(debug=True, server_port=8002)