"""
This code is largely based on https://github.com/jshilong/GPT4RoI/blob/main/gpt4roi/datasets/vcr.py
"""
import copy
import json
import os
import random
from tkinter import N

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import path
from matplotlib import pyplot as plt
from osprey.train.train import preprocess, preprocess_multimodal

WHY_QUESTIONS = [
    'why?',
    'why',
    "What's the rationale for your decision?",
    'What led you to that conclusion?',
    "What's the reasoning behind your opinion?",
    'Why do you believe that to be true?',
    'Can you explain the basis for your thinking?',
    'What factors influenced your perspective?',
    'How did you arrive at that perspective?',
    'What evidence supports your viewpoint?',
    'What makes you think that way?',
    "What's the logic behind your argument?",
    'Can you provide some context for your opinion?',
    "What's the basis for your assertion?",
    'Why do you hold that belief?',
    'What experiences have shaped your perspective?',
    'What assumptions underlie your reasoning?',
    "What's the foundation of your assertion?",
    "What's the source of your reasoning?",
    "What's the motivation behind your decision?",
    "What's the impetus for your belief?",
    "What's the driving force behind your conclusion?",
    'Why do you think that?',
    "What's your reasoning?",
    'What makes you say that?',
    'Why do you feel that way?',
    "What's the story behind that?",
    "What's your thought process?",
    "What's the deal with that?",
    "What's the logic behind it?",
    'Why do you believe that?',
    "What's the real deal here?",
    "What's the reason behind it?",
    "What's the thought process behind your decision?",
    "What's the rationale for your opinion?",
    'Why do you have that impression?',
    "What's the background to that?",
    "What's the evidence that supports your view?",
    "What's the explanation for that?"
]

Ref_WAY = [
    'There are <region> in the image,',
    'There are some regions <region>,',
    'Given <region>,',
    'Given <region> in the image,',
    '<region>,',
    'Several regions <region> are in the image,',
    '<region> in the given image,'
]

def _spaced_points(low, high,n):
    """ We want n points between low and high, but we don't want them to touch either side"""
    padding = (high-low)/(n*2)
    return np.linspace(low + padding, high-padding, num=n)

def make_mask(height, width, box, polygons_list):
    """
    Mask size: int about how big mask will be
    box: [x1, y1, x2, y2, conf.]
    polygons_list: List of polygons that go inside the box
    """
    mask = np.zeros((height, width), dtype=np.bool_)
    
    xy = np.meshgrid(_spaced_points(box[0], box[2], n=width),
                     _spaced_points(box[1], box[3], n=height)) 
    xy_flat = np.stack(xy, 2).reshape((-1, 2))

    for polygon in polygons_list:
        polygon_path = path.Path(polygon)
        mask |= polygon_path.contains_points(xy_flat).reshape((height, width))
    return mask.astype(np.float32)

class VCRDataset(Dataset):
    CLASSES = ('object',)

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,

                 ):
        super(VCRDataset, self).__init__()


        self.img_prefix = img_prefix

        self.tokenizer = tokenizer

        self.data_args = data_args

        self.begin_str = """<image>.\nThis provides an overview of the picture.\n"""
        self.data_infos = self.load_annotations(ann_file)
        print('normal_vcr', len(self.data_infos))

    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
          ann_list = [json.loads(line) for line in f]
        data_infos = []

        import re

        def replace_numbers_with_tags(s, class_names):
            pattern = r'\b(\d+)\b'
            try:
                result = re.sub(pattern, lambda match: f'{class_names[int(match.group(1))]} at region{match.group(1)}', s)
            except:
                # contain number not for instance
                return None
            return result


        for ann in ann_list:

            metadata_fn_path = ann['metadata_fn']
            img_fn = ann['img_fn']
            img_path = os.path.join(self.img_prefix,img_fn)
            annotations = json.load(open(os.path.join(self.img_prefix, metadata_fn_path)))
            masks = annotations['segms']
            bboxes = np.array(annotations['boxes'])

            class_names = ann['objects']
            num_objects = len(class_names)
            ref_string = ''
            for i in range(num_objects):
                ref_string = ref_string +  f'region{i+1} <mask><pos>' + ','
            ref_string = ref_string[:-1]
            ref_prefix = random.choice(Ref_WAY)

            begion_string = ref_prefix.replace('<region>', ref_string)
            qa_s = []

            q = ann['question_orig']
            q = replace_numbers_with_tags(q, class_names)
            a = ann['answer_orig']
            a = replace_numbers_with_tags(a, class_names)
            why = ann['rationale_orig']
            why = replace_numbers_with_tags(why, class_names)
            if (q is None) or (a is None) or (why) is None:
                continue


            qa_s.append({'from': 'human', 'value': begion_string + q})
            qa_s.append({'from': 'gpt', 'value': a})
            qa_s.append({'from': 'human', 'value': random.choice(WHY_QUESTIONS)})
            qa_s.append({'from': 'gpt', 'value': why})

            data_infos.append(dict(
                img_path = img_path,
                bboxes = bboxes,
                masks = masks,
                labels= class_names,
                qas = qa_s)
            )


        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, i):
        data_info = self.data_infos[i]

        img_path = data_info['img_path']
        masks = data_info['masks']
        bboxes = data_info['bboxes']

        qas = data_info['qas']
        processor = self.data_args.image_processor
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        # TODO ablation this

        image_file = img_path

        pred_masks = np.zeros((len(masks), h, w))
        for i,mask in enumerate(masks):

            int_box =  [round(box) for box in bboxes[i][:-1]]
            
            height_ = int(int_box[3]-int_box[1])
            width_ = int(int_box[2]-int_box[0])
            box_mask = make_mask(height_, width_, bboxes[i], mask)

            pred_masks[i, int_box[1]:int_box[3], int_box[0]:int_box[2]] = box_mask

        image = processor.preprocess(image,
                                     do_center_crop=False,
                                     return_tensors='pt')['pixel_values'][0]

        image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(512, 512),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)

        cur_token_len = (image.shape[1] // 16) * (image.shape[2] // 16)  # FIXME: 16 is hardcoded patch size
        qas = copy.deepcopy(qas)
        qas[0]['value'] = self.begin_str + qas[0]['value']

        sources = preprocess_multimodal(
            copy.deepcopy([qas]),
            self.data_args, cur_token_len)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image
        data_dict['masks'] = torch.Tensor(pred_masks)

        return data_dict
