import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from torch import nn
import copy
from functools import partial
from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.utils import disable_torch_init
from osprey.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from osprey.train.train import preprocess, preprocess_multimodal
from osprey.train.train import DataArguments
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from pycocotools import mask as maskUtils
import numpy as np
from PIL import Image
import cv2
import re
data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

def annToMask(ann, h, w):
    rles = maskUtils.frPyObjects(ann, h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m

class GPT_EVAL(nn.Module):
    def __init__(self, model_path, model_base=None):
        super().__init__()
        disable_torch_init()
        model_path = os.path.expanduser(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=True
        )
        self.model = OspreyLlamaForCausalLM.from_pretrained(
                                                model_path,
                                                torch_dtype=torch.bfloat16,
                                                ).cuda()
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge":512}, resample=3,  do_center_crop=True, crop_size={"height": 512, "width": 512},
                                                  do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                  image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        
        spi_tokens = ['<mask>', '<pos>']
        self.tokenizer.add_tokens(spi_tokens, special_tokens=True)
        
        for m in self.model.modules():
            m.tokenizer = self.tokenizer

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.float16, device='cuda')
        

    def forward(self, root_path, ann_file):
        final = []
        anns = json.load(open(ann_file))
        
        for i, ann in enumerate(anns):
            print(i)

            model_answer = {}
            model_answer["question_id"] = ann["question_id"]
            model_answer["image"] = ann["image"]
            model_answer["category"] = ann["category"]
            img_path = os.path.join(root_path, ann['image'])
            img = cv2.imread(img_path)
            question = ann['text']
        
            question = re.sub(r'<region>', r'<mask><pos>', question)
            # question += 'Answer the question in detail.'
            idx = 1
         
            mask_r = ann['annotation']['segmentation']
            height, width = img.shape[:2]

            if isinstance(mask_r, list):
                mask = annToMask(mask_r, height, width)
            else:
                mask = maskUtils.decode(mask_r)
            mask = torch.from_numpy(mask).unsqueeze(0)

            x1, y1, w, h = ann['annotation']['bbox']
            bbox = np.array([x1, y1, x1 + w, y1 + h])
            bbox = torch.from_numpy(bbox)

        
            init_inputs = get_init_inputs(img_path,
                                        self.image_processor,
                                        self.tokenizer,
                                        pred_bboxes=bbox,
                                        mask=mask,
                                        question=question,
                                        round_ids=0,
                                        last_round_source={},
                                        )

            masks = init_inputs['masks'].cuda()
            image = init_inputs['image']
            conv = conv_templates['osprey_v1'].copy()
            qs = init_inputs['sources'][0][0]['value']
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            self.model.model.tokenizer = self.tokenizer


            with torch.inference_mode():

                self.model.orig_forward = self.model.forward
                self.model.forward = partial(self.model.orig_forward,
                                             img_metas=[None],
                                             masks=[masks.half()])

                output_ids = self.model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    # masks=[masks.half()],
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    num_beams=1,
                    # stopping_criteria=[stopping_criteria]
                    )

                self.model.forward = self.model.orig_forward

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(
                    f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                                  skip_special_tokens=True)[0]

            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            model_answer['text'] = outputs
            
            print(outputs)
            final.append(model_answer)

        final_ = json.dumps(final)
        with open('ferret_bench/osprey_refer_reason_original_3.json','w') as fw:
            fw.write(final_)
            fw.close()
     

def get_init_inputs(img_path,
                    processor,
                    tokenizer,
                    pred_bboxes,
                    mask,
                    question=None,
                    round_ids=0,
                    last_round_source=None):

    if round_ids == 0:
       
        image = Image.open(img_path).convert('RGB')

        image = processor.preprocess(image,
                                     do_center_crop=False,
                                     return_tensors='pt')['pixel_values'][0]

        image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(512, 512),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)

    else:
        image = last_round_source['image']

    cur_token_len = (image.shape[1] // 16) * (image.shape[2] // 16)

    mask = mask.to(image.device)

    begin_str = """<image>.\nThis provides an overview of the picture.\n"""

    sources = dict()
    sources['conversations'] = []

    sources['conversations'].append({'from': 'human', 'value': begin_str+question})
    
    sources = preprocess_multimodal([sources['conversations']], data_args, cur_token_len)

    data_dict = {}
    data_dict['sources'] = sources
    data_dict['image'] = image
    data_dict['bboxes'] = pred_bboxes
    data_dict['masks'] = mask
    data_dict['img_metas'] = dict(filename=img_path)

    return data_dict


if __name__ == "__main__":
    model_name = '/Osprey-Chat-7b'
    root_path = '/path/to/coco-imgs'
    json_path = './ferret_bench/refer_reason/box_refer_reason.json'
    ferret_eval = GPT_EVAL(model_name)
    ferret_eval(root_path, json_path)

