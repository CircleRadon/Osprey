import argparse
import torch
import os
import json
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import preprocess_multimodal
from osprey.train.train import DataArguments
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
from PIL import Image
data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

def annToMask(ann, h, w):
    rles = maskUtils.frPyObjects(ann, h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m

class REFCOCOG_EVAL():
    def __init__(self, model_path,):
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

    def forward(self, root_path, ann_file, gt_file='captions_refcocog_gt.json', caption_file='captions_refcocog_osprey.json'):
        self.captions_all = []
        self.gt_all = {}
        self.gt_all['images'] = []
        self.gt_all['annotations'] = []
        self.root_path = root_path
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        for i, img in enumerate(tqdm(self.img_ids)):
            data = self.coco.loadImgs([img])[0]
            self.forward_single(data)

        final = json.dumps(self.captions_all)
        with open(caption_file,'w') as fw:
            fw.write(final)
            fw.close()
        
        final = json.dumps(self.gt_all)
        with open(gt_file,'w') as fw:
            fw.write(final)
            fw.close()

   
    def forward_single(self, inputs):
    
        img_path = os.path.join(self.root_path, inputs['file_name'].split('_')[-1])
        height = inputs['height']
        width = inputs['width']
        round_ids = 0
        last_source = dict()
        annotations_ids = self.coco.getAnnIds([inputs['id']])
        annotations = self.coco.loadAnns(annotations_ids)
        for i in range(len(annotations)):
            caption = {}
            gt = {}
            ann = annotations[i]
            mask_r = ann['segmentation']

            if isinstance(mask_r, list):
                mask = annToMask(mask_r, height, width)
            else:
                mask = maskUtils.decode(mask_r)
            mask = torch.from_numpy(mask).unsqueeze(0)
        
            init_inputs = get_init_inputs(img_path,
                                        self.image_processor,
                                        self.tokenizer,
                                        mask=mask,
                                        round_ids=round_ids,
                                        last_round_source=last_source,
                                        )

            round_ids += 1
            last_source = init_inputs

            image = init_inputs['image']

            masks = init_inputs['masks'].cuda()

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
            
            print(outputs)
            outputs = outputs.replace('.', '.\n')
            caption['image_id'] = str(ann['id'])
            caption['caption'] = outputs
            gt['id'] = str(ann['id'])
            gt['image_id'] = str(ann['id'])
            gt['caption'] = inputs['caption']

            self.captions_all.append(caption)
            self.gt_all['annotations'].append(gt)
            self.gt_all['images'].append({'id':str(ann['id'])})



def get_init_inputs(img_path,
                    processor,
                    pred_bboxes,
                    mask,
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
    question = 'Please give me a short description of <mask><pos>.'

    sources['conversations'].append({'from': 'human', 'value': begin_str+question})
    
    sources = preprocess_multimodal([sources['conversations']], data_args, cur_token_len)

    data_dict = {}
    data_dict['sources'] = sources
    data_dict['image'] = image
    data_dict['masks'] = mask

    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='osprey demo', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', default='path/to/Osprey-7B-refcocog-fintune')
    parser.add_argument('--img', help='path to coco imgs', default='path/to/coco_all_imgs/')
    parser.add_argument('--json', help='path to refcocog val json file', default='./finetune_refcocog_val_with_mask.json')
    args = parser.parse_args()

    refcocog_eval = REFCOCOG_EVAL(args.model)
    refcocog_eval.forward(args.img, args.json)

