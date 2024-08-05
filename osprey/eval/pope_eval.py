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
import numpy as np
from PIL import Image
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True


class POPE_EVAL():
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
        

    def eval(self, root_path, ann_file, answer_file):
        data_all = [json.loads(l) for l in open(ann_file, 'r')]
        ans_file = open(answer_file, 'w')
        
        for data in tqdm(data_all):
            try:
                img_path = os.path.join(root_path, data['image'])
                image = Image.open(img_path).convert('RGB')
            except:
                img_path = os.path.join(root_path, data['image'].split('_')[-1])
                image = Image.open(img_path).convert('RGB')
            
            init_inputs = get_init_inputs(image,
                                        self.image_processor,
                                        data['text']
                                        )

            image = init_inputs['image']

            conv = conv_templates['osprey_v1'].copy()
            qs = init_inputs['sources'][0][0]['value']

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            self.model.model.tokenizer = self.tokenizer

            with torch.inference_mode():

                self.model.orig_forward = self.model.forward
                self.model.forward = partial(self.model.orig_forward,
                                            img_metas=[None],
                                            )

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

            
            ans_file.write(json.dumps({"question": data['text'],
                                       "answer": outputs.lower()})+"\n")
           

def get_init_inputs(image,
                    processor,
                    input_question):

    image = processor.preprocess(image,
                                    do_center_crop=False,
                                    return_tensors='pt')['pixel_values'][0]

    image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                            size=(512, 512),
                                            mode='bilinear',
                                            align_corners=False).squeeze(0)


    cur_token_len = (image.shape[1] // 16) * (image.shape[2] // 16)

    sources = dict()
    sources['conversations'] = []

    question = '<image>\n'+input_question


    sources['conversations'].append({'from': 'human', 'value': question})
    
    sources = preprocess_multimodal([sources['conversations']], data_args, cur_token_len)

    data_dict = {}
    data_dict['sources'] = sources
    data_dict['image'] = image
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='osprey demo', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', default='osprey-7b')
    parser.add_argument('--img', help='path to coco imgs', default='/path/to/coco-imgs')
    parser.add_argument('--json', help='path to pope val json file', default='pope/coco_pope_random.json') #'pope/coco_pope_adversarial.json', 'pope/coco_pope_popular.json', 'pope/coco_pope_random.json'
    parser.add_argument('--answer', help='path to answer json file', default='./osprey_pope_random_answer.json')

    args = parser.parse_args()

    POPE_EVAL = POPE_EVAL(args.model)
    POPE_EVAL.eval(args.img, args.json, args.answer)

