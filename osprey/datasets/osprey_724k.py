import numpy as np
import torch
import json
import os
import copy
import random
from .stage2_data import CustomDataset
from osprey.train.train import preprocess, preprocess_multimodal
import re

DETAILED_QUESTIONS =  [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give ablout the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represtented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]


class ConversationDataset(CustomDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        self.begin_str = "<image>\nThis provides an overview of the picture.\n"
        super().__init__(tokenizer, data_args, ann_file, img_prefix)


    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            if len(ann['conversations'])//2 ==0:
                continue
            masks = []
            qa_s = []
            filename = ann['file_name'].split('_')[-1]
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']
            str_region = ""
            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)
                if i>0:
                    str_region+= ','
                str_region += "region"+str(i+1)+"<mask><pos>"

            for i in range(len(ann['conversations'])//2):
                    
                if i==0:
                    if region_num==1:
                        mid_str = "Ther are 1 part region in the picture: "+str_region+'. '
                    else:
                        mid_str = "Ther are {} part regions in the picture: ".format(str(region_num))+str_region+'. '

                    question = ann['conversations'][i*2]['value']
                    question = question.replace('<','').replace('>','')
                    question = self.begin_str+mid_str+question
                    qa_s.append({'from': 'human', 'value': question+self.limit})         
                else:
                    question = ann['conversations'][i*2]['value']
                    question = question.replace('<','').replace('>','')
                    qa_s.append({'from': 'human', 'value': question+self.limit})         

                
                answer = ann['conversations'][i*2+1]['value']
                answer = answer.replace('<','').replace('>','')
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
        return data_infos

    def __getitem__(self, i):
        data_info = self.data_infos[i]
        img_path = data_info['img_path']
        height = data_info['height']
        width = data_info['width']
        masks_raw = data_info['masks']
        masks = []
        for mask_r in masks_raw:
            mask = self.annToMask(mask_r, height, width)
            masks.append(mask)
            
        masks = np.array(masks)
        qas = data_info['qas']

        image = self.read_process_image(img_path)
       
        cur_token_len = (image.shape[1] // 16) * (image.shape[2] // 16)  # FIXME: 16 is hardcoded patch size

        sources = preprocess_multimodal(
            copy.deepcopy([qas]),
            self.data_args, cur_token_len)
        # print(sources)
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image

        data_dict['masks'] = torch.Tensor(masks)

        return data_dict

class OspreyPartLevel(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(tokenizer, data_args, ann_file, img_prefix)

class OspreyLVISPosNeg(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        
        super().__init__(tokenizer, data_args, ann_file, img_prefix)

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            if len(ann['conversations'])//2 ==0:
                continue
            masks = []
            qa_s = []
            filename = ann['file_name']
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']

            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)
        
            for i in range(len(ann['conversations'])//2):
                    
                question = ann['conversations'][i*2]['value']
                question = re.sub(r'<region\d+>', '<mask><pos>', question)
                if i==0:
                    question = self.begin_str+question
                qa_s.append({'from': 'human', 'value': question})         
             
                answer = ann['conversations'][i*2+1]['value']
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
            # print(qa_s)

        return data_infos

      

class OspreyConversations(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        self.limit = ""
        super().__init__(tokenizer, data_args, ann_file, img_prefix)

class OspreyShortForm(ConversationDataset):
     def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(tokenizer, data_args, ann_file, img_prefix)

class OspreyDetailedDescription(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 ):
        super().__init__(tokenizer, data_args, ann_file, img_prefix)

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            masks = []
            qa_s = []
            filename = ann['file_name'].split('_')[-1]
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']
            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)

                question = random.choice(DETAILED_QUESTIONS)
                question = question.replace('<region>', '<mask><pos>')
                if i==0:
                    qa_s.append({'from': 'human', 'value': self.begin_str+question})         
                else:
                    qa_s.append({'from': 'human', 'value': question})     
            
                answer = re.findall(r"<.*>:\ (.*)", ann['description'][i])[0]
           
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
        return data_infos
