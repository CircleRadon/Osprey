import argparse
import torch
import os
import json
from functools import partial
from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import preprocess_multimodal
from osprey.train.train import DataArguments
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from pycocotools import mask as maskUtils
import numpy as np
from PIL import Image
import cv2

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

def annToMask(ann, h, w):
    rles = maskUtils.frPyObjects(ann, h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m

class GPT_EVAL():
    def __init__(self, model_path):
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
        

    def eval(self, root_path, ann_file):
        final = []
        anns = json.load(open(ann_file))
        
        for i, ann in enumerate(anns):
            print(i)

            model_answer = {}
            model_answer["question_id"] = ann["question_id"]
            model_answer["image"] = ann["image"]
            model_answer["category"] = ann["category"]
            img_path = os.path.join(root_path, ann["image"])
            img = cv2.imread(img_path)
            question = ann["text"]
         
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
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    num_beams=1,
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
            if ':' in outputs:
                outputs = outputs.split(':')[1]

            model_answer['text'] = outputs
            
            print(outputs)
            final.append(model_answer)

        final_ = json.dumps(final)
        with open('description/osprey_answer.json','w') as fw:
            fw.write(final_)
            fw.close()
     

def get_init_inputs(img_path,
                    processor,
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
    question = question.replace('<region>','<mask><pos>')

    sources['conversations'].append({'from': 'human', 'value': begin_str+question})
    
    sources = preprocess_multimodal([sources['conversations']], data_args, cur_token_len)
   
    data_dict = {}
    data_dict['sources'] = sources
    data_dict['image'] = image
    data_dict['masks'] = mask

    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='osprey generate gpt answer', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', default='/path/to/osprey-7b')
    parser.add_argument('--coco-img', help='path to coco imgs', default='/path/to/coco_all_imgs/')
    parser.add_argument('--json', help='path to question json file', default='./description/questions.json')
    args = parser.parse_args()

    gpt_eval = GPT_EVAL(args.model)
    gpt_eval.eval(args.coco_img, args.json)
