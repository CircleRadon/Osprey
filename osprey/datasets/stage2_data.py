
import copy
import os
import random
import numpy as np
import torch

from osprey.train.train import preprocess, preprocess_multimodal
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image

class CustomDataset(Dataset):

    def __init__(self,
                 tokenizer=None,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=20,
                 ):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.max_gt_per_img = max_gt_per_img
        self.img_prefix = img_prefix

        self.data_infos = self.load_annotations(ann_file)
        super().__init__()

    def __len__(self):
        return len(self.data_infos)
    
    def load_annotations(self, ann_file):

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]

            info['filename'] = info['file_name']
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info)==0:
                continue

            data_infos.append(info)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
    
    def get_ann_info(self, idx):

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return ann_info
    
    def annToMask(self, mask_ann, h, w):
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_text(self, data_item):
        image = data_item['img']
        ori_labels = data_item['gt_labels']
        ori_masks = np.array(data_item['gt_masks'])
        ori_masks = torch.from_numpy(ori_masks) 

        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        ori_masks = ori_masks[shuffle_ids]
        ori_labels = [ori_labels[i] for i in shuffle_ids]

        sources = dict()

        sources['conversations'] = []

        # print("num:",len(ori_labels))

        for i in range(len(ori_labels)):
            question = '<region>'
            question = question.replace('<region>', '<mask><pos>')
            if i == 0:
                question = self.begin_str + question
            answer = ori_labels[i]
            sources['conversations'].append(
                {'from': 'human', 'value': question})
            sources['conversations'].append({'from': 'gpt', 'value': answer})

        cur_token_len = (image.shape[1] // 16) * (image.shape[2] // 16)

        assert image.shape[1] == image.shape[2]
        # a hard code [] for sources
        sources = preprocess_multimodal(
            copy.deepcopy([sources['conversations']]),
            self.data_args,
            cur_token_len)
        # print(sources)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True
            )
        
        # get single
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image
        data_dict['masks'] = ori_masks
        return data_dict

    def read_process_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        
        processor = self.data_args.image_processor

        image = processor.preprocess(image,
                                     do_center_crop=False,
                                     return_tensors='pt')['pixel_values'][0]

        image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(512, 512),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)
        return image
    
    def get_data_item(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        img_path =os.path.join(self.img_prefix, data_info['filename'])
        image = self.read_process_image(img_path)

        gt_masks = []
        gt_labels = []
        for ann in ann_info:
            mask = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            gt_masks.append(mask)

            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(cat[0]['name'])

        data_item = dict(
            img = image,
            gt_masks = gt_masks,
            gt_labels = gt_labels
        )
        return data_item

    def __getitem__(self, idx):

        data_item = self.get_data_item(idx)
        data_dict = self.process_text(data_item=data_item)

        return data_dict

class COCODataset(CustomDataset):

    def __init__(self,
                 tokenizer=None,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=20,
                 ):

        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img)
        self.begin_str = '<image>\nIn the conversation below, you simply answer the category name based on what you see ' \
                        'in the imagery inside a particular region. I will give you only one region each time.\n' 

class PartImagenet(CustomDataset):

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 ):

        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img)
        CAT_CLASSES = (
            'Bottle', 'Biped', 'Quadruped', 'Fish', 'Reptile', 'Bicycle', 'Bird', 'Car', 'Boat', 'Snake', 'Aeroplane'
        )

        SUB_CLASSES = (
            'Tier', 'Hand', 'Wing', 'Mouth', 'Tail', 'Side', 'Fin', 'Engine', 'Foot', 'Head', 'Body', 'Sail', 'Seat'
        )

        begin_str = '<image>\nIn the conversation below, you simply answer the category and subcategory name based on what you see' \
                            'in the image inside a particular region. It maybe a subpart of an object. '\
                            'I will give you only one region each time. Your answer should in the format of '\
                            'category subcategory. '
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        self.begin_str = begin_str + class_str + subclass_str + '.\n'

class PascalPart(CustomDataset):

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 ):

        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img)
        CAT_CLASSES = ('potted plant', 'aeroplane', 'cow', 'cat', 'bus', 'horse', 'car', 
                    'dog', 'bicycle', 'person', 'bird', 'bottle', 'sheep', 'motorbike')

        SUB_CLASSES = ('eye', 'window', 'cap', 'headlight', 'hand', 'mirror', 'arm', 'plant', 
                    'wheel', 'ear', 'pot', 'foot', 'leg', 'nose', 'body', 'horn', 'handlebar', 
                    'neck', 'license plate', 'paw', 'saddle', 'head', 'muzzle', 'tail', 'wing', 
                    'beak', 'hair', 'torso', 'door', 'mouth')

        begin_str = '<image>\n In the conversation below, you simply answer the category and subcategory name based on what you see' \
                            'in the image inside a particular region. It maybe a subpart of an object. '\
                            'I will give you only one region each time. Your answer should in the format of '\
                            'category:subcategory. '
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        self.begin_str = begin_str + class_str + subclass_str + '.\n'

class RefCOCO(CustomDataset):

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 ):

        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img)

        self.begin_str = '<image>\nI will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image, as well as its position within ' \
                         'the image and its basic attributes.'

    def load_annotations(self, ann_file):

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]

            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info)==0:
                continue
            
            data_infos.append(info)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
        
    def get_data_item(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        img_path =os.path.join(self.img_prefix, data_info['filename'])
        image = self.read_process_image(img_path)

        gt_masks = []
        gt_labels = []
        for ann in ann_info:
            mask = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            gt_masks.append(mask)
            
            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(data_info['caption'])

        data_item = dict(
            img = image,
            gt_masks = gt_masks,
            gt_labels = gt_labels
        )
        return data_item

class RefCOCOP(RefCOCO):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 ):
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img)
        self.begin_str = '<image>\nI will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image and its basic attibuts, you should not ' \
                         'give its position within the image.'                        
