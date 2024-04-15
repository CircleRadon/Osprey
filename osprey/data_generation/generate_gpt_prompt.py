from coco_api import COCO, COCOeval
import json

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
class COCODataset():
    def __init__(self, annotation_file='data/instances_train2017.json'):
        
        self.coco = COCO(annotation_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
    
    def get_annotations(self, img_ids, width, height):
        ann_ids = self.coco.get_ann_ids(img_ids=[img_ids])
        anns = self.coco.load_anns(ann_ids)
        anns = self.simplify(anns, width, height)
        # gpt_format = self.to_gpt_format(anns)
        return anns

    def simplify(self, anns, width, height):
        sim_anns = []
        for ann in anns:
            new_ann = {}
            new_ann['bbox'] = ann['bbox'].copy()
            new_ann['bbox'][0] = ann['bbox'][0] / width
            new_ann['bbox'][1] = ann['bbox'][1] / height
            new_ann['bbox'][2] = (ann['bbox'][0]+ann['bbox'][2]) / width
            new_ann['bbox'][3] = (ann['bbox'][1]+ann['bbox'][3]) / height
            new_ann['category'] = CLASSES[self.cat2label[ann['category_id']]]
            new_ann['img_id'] = ann['image_id']
            sim_anns.append(new_ann)
        return sim_anns

class COCOCaption():
    def __init__(self, annotation_file='data/captions_train2017.json'):
        self.caption = json.load(open(annotation_file))
        self.imgid2id = {}
        for i in range(len(self.caption['annotations'])):
            if self.imgid2id.get(self.caption['annotations'][i]['image_id']) is None:
                self.imgid2id[self.caption['annotations'][i]['image_id']] = [i]
            else:
                self.imgid2id[self.caption['annotations'][i]['image_id']].append(i)
    
    def get_annotations(self, img_ids):
        ann_ids = self.imgid2id[img_ids]
        captions = []
        for id in ann_ids:
            captions.append(self.caption['annotations'][id]['caption'])
        return captions

class RefCOCO():
    def __init__(self, annotation_file='data/finetune_refcoco_train_with_mask.json'):
        self.coco = COCO(annotation_file)
        self.raw_id_to_img_id = {}
        self.cat_ids = self.coco.get_cat_ids(cat_names=CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        for id in self.coco.imgs:
            img = self.coco.imgs[id]
            if img['original_id'] in self.raw_id_to_img_id.keys():
                self.raw_id_to_img_id[img['original_id']].append(img['id'])
            else:
                self.raw_id_to_img_id[img['original_id']] = []
                self.raw_id_to_img_id[img['original_id']].append(img['id'])
    
    def get_annotations_and_captions(self, raw_ids):
        if raw_ids not in self.raw_id_to_img_id.keys():
            return None
        img_ids = self.raw_id_to_img_id[raw_ids]
        infos = self.coco.load_imgs(img_ids)
        ann_ids = self.coco.get_ann_ids(img_ids=img_ids)
        anns = self.coco.load_anns(ann_ids)
        anns = self.simplify(anns, infos)
        return anns
    
    def simplify(self, anns, infos):
        sim_anns = []
        for ann, info in zip(anns, infos):
            new_ann = {}
            new_ann['bbox'] = ann['bbox']
            new_ann['segmentation'] = ann['segmentation']
            new_ann['category'] = CLASSES[self.cat2label[ann['category_id']]]
            new_ann['img_id'] = ann['image_id']
            new_ann['caption'] = info['caption']
            new_ann['original_id'] = ann['original_id']
            sim_anns.append(new_ann)
        return sim_anns
    

class RefCOCOP(RefCOCO):
    def __init__(self, annotation_file='data/finetune_refcoco+_train_with_mask.json'):
        self.coco = COCO(annotation_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        self.raw_id_to_img_id = {}
        for id in self.coco.imgs:
            img = self.coco.imgs[id]
            if img['original_id'] in self.raw_id_to_img_id.keys():
                self.raw_id_to_img_id[img['original_id']].append(img['id'])
            else:
                self.raw_id_to_img_id[img['original_id']] = []
                self.raw_id_to_img_id[img['original_id']].append(img['id'])

class RefCOCOg(RefCOCO):
    def __init__(self, annotation_file='data/finetune_refcocog_train_with_mask.json'):
        self.coco = COCO(annotation_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        self.raw_id_to_img_id = {}
        for id in self.coco.imgs:
            img = self.coco.imgs[id]
            if img['original_id'] in self.raw_id_to_img_id.keys():
                self.raw_id_to_img_id[img['original_id']].append(img['id'])
            else:
                self.raw_id_to_img_id[img['original_id']] = []
                self.raw_id_to_img_id[img['original_id']].append(img['id'])

def anns_to_gpt_format(anns):
    qs = ""
    for ann in anns:
        qs = qs+ ann['category'] + ": " + box_tostr(ann["bbox"]) + '\n'
    return qs

def change_box(boxes, width, height):
    ret_boxes = []
    for box in boxes:
        new_box = box.copy()
        new_box[0] = box[0] / width
        new_box[1] = box[1] / height
        new_box[2] = (box[0]+box[2]) / width
        new_box[3] = (box[1]+box[3]) / height
        ret_boxes.append(new_box)
    return ret_boxes

def box_tostr(box):
    box_str = ""
    box_str +="[" + str('%.3f'%box[0]) + "," + str('%.3f'%box[1]) + "," + str('%.3f'%box[2]) + "," + str('%.3f'%box[3]) + "]"
    return box_str

def caption_to_gpt_format(all_boxes, cap_idx, captions, categories):
    qs = ""
    for i in range(len(cap_idx)):
        idx = cap_idx[i]
        qs += "For <regin{}>".format(i+1) + "("+categories[i]+": "
        qs += box_tostr(all_boxes[i]) + "):" + '\n'
        for caption in captions[idx]:
            qs += caption + '\n'
        qs += '\n'
    return qs

def caption_to_gpt_format1(all_boxes, cap_idx, captions, categories):
    qs = ""
    for i in range(len(cap_idx)):
        idx = cap_idx[i]
        qs += "<regin{}>".format(i+1) + "("+categories[i] + "):" + '\n'
        for caption in captions[idx]:
            qs += caption + '\n'
        qs += '\n'
    return qs

class LLaVA():
    def __init__(self, annotation_file='data/detail_23k.json'):
        self.data = json.load(open(annotation_file))
        self.raw_id_to_llava_id = {}
        for i in range(len(self.data)):
            item = self.data[i]
            self.raw_id_to_llava_id[int(item['id'])] = i
    
    def get_detail_description(self, raw_id):
        if self.raw_id_to_llava_id.get(raw_id) is None:
            return None
        llava_id = self.raw_id_to_llava_id[raw_id]
        return self.data[llava_id]['conversations'][1]['value']

def generate_region_str(num_boxes):
    region_str = ''
    for i in range(num_boxes):
        region_str += '<region{}>'.format(i+1)
        if i<num_boxes-2:
            region_str += ', '
        elif i==num_boxes-2:
            region_str += ' and '
    return region_str

def generate_gpt_prompt_cat(llava_desc, coco_gpt_format, caption_gpt_format, num_boxes, coco_cap_anns):
    prompt = ''
    if llava_desc is not None:
        prompt += 'The detailed description of this image:\n' + '"' + llava_desc + '"' + '\n'
    else:
        prompt += 'Several descriptions of this image:\n'
        for coco_cap in coco_cap_anns:
            prompt += coco_cap + '\n'
    prompt += '\n'
    mid_str = 'Specially, there are {} special regions: '.format(str(num_boxes))
    region_str = generate_region_str(num_boxes)
    mid_str += region_str +'.'
    prompt += mid_str+'\n'
    prompt += caption_gpt_format
    return prompt

def generate_gpt_prompt(llava_desc, coco_gpt_format, caption_gpt_format, num_boxes, coco_cap_anns):
    prompt = ''
    if llava_desc is not None:
        prompt += 'The detailed description of this image:\n' + '"' + llava_desc + '"' + '\n'
    else:
        prompt += 'Several descriptions of this image:\n'
        for coco_cap in coco_cap_anns:
            prompt += coco_cap + '\n'
    prompt += '\n'+coco_gpt_format
    mid_str = 'Specially, there are {} special regions: '.format(str(num_boxes))
    region_str = generate_region_str(num_boxes)
    mid_str += region_str +'.'
    mid_str += ' For each one region, you receive several sentences as the description of this region in this image you are observing.'
    prompt += mid_str+'\n'
    prompt += caption_gpt_format
    return prompt

def generate_gpt_prompt_qa(llava_desc, coco_gpt_format, caption_gpt_format, num_boxes, coco_cap_anns):
    prompt = ''
    if llava_desc is not None:
        prompt += 'Whole description:\n' + '"' + llava_desc + '"' + '\n'
    else:
        prompt += 'Several descriptions of the whole image:\n'
        for coco_cap in coco_cap_anns:
            prompt += coco_cap + '\n'
    prompt+='\n'+'Description of each region is listed below:\n'
    prompt+=caption_gpt_format
    return prompt


class GeneratePrompt():
    def __init__(self):
        self.coco = COCODataset()
        self.refcoco = RefCOCO()
        self.refcocop = RefCOCOP()
        self.refcocog = RefCOCOg()
        self.cococap = COCOCaption()
        self.llava = LLaVA()

    def load_data_and_generate_gpt_prompt_category(self, raw_img_id, type=0):
        llava_desc = None

        coco_cap_anns = self.cococap.get_annotations(raw_img_id)
      
        img_info = self.coco.coco.load_imgs([raw_img_id])[0]
        height = img_info['height']
        width = img_info['width']
        anns = self.coco.get_annotations(raw_img_id, width, height)
        coco_gpt_format = anns_to_gpt_format(anns)

        ref_anns_all = []
        refcoco_anns = self.refcoco.get_annotations_and_captions(raw_img_id)
        if refcoco_anns is not None:
            ref_anns_all.extend(refcoco_anns)

        refcocop_anns = self.refcocop.get_annotations_and_captions(raw_img_id)
        if refcocop_anns is not None:
            ref_anns_all.extend(refcocop_anns)

        refcocog_anns = self.refcocog.get_annotations_and_captions(raw_img_id)
        if refcocog_anns is not None:
            ref_anns_all.extend(refcocog_anns)

        if len(ref_anns_all)==0:
            return None, None, None, None, None, None

        captions = {}
        cap_idx = []
        all_boxes = []
        categories = []
        annotations = []

        for ann in ref_anns_all:
            id = ann['original_id']
            if captions.get(id) is None:
                captions[id] = {ann['caption']}
                cap_idx.append(id)
                all_boxes.append(ann['bbox'])
                categories.append(ann['category'])
                new_ann = {}
                new_ann['bbox'] = ann['bbox']
                new_ann['segmentation'] = ann['segmentation']
                annotations.append(new_ann)
            else:
                captions[id].add(ann['caption'])

        all_boxes = change_box(all_boxes, width, height)

        num_boxes = len(all_boxes)

        caption_gpt_format = caption_to_gpt_format(all_boxes, cap_idx, captions, categories)

        gpt_format = generate_gpt_prompt_cat(llava_desc, coco_gpt_format, caption_gpt_format, num_boxes, coco_cap_anns)
        
        return gpt_format, annotations, num_boxes, height, width

        

    def load_data_and_generate_gpt_prompt_description(self, raw_img_id, type=0):
        # get llava description
        llava_desc = self.llava.get_detail_description(raw_img_id)

        if llava_desc is None:
            coco_cap_anns = self.cococap.get_annotations(raw_img_id)
        else:
            coco_cap_anns = None

        # get coco gpt format
        img_info = self.coco.coco.load_imgs([raw_img_id])[0]
        height = img_info['height']
        width = img_info['width']
        anns = self.coco.get_annotations(raw_img_id, width, height)
        coco_gpt_format = anns_to_gpt_format(anns)

        ref_anns_all = []
        refcoco_anns = self.refcoco.get_annotations_and_captions(raw_img_id)
        if refcoco_anns is not None:
            ref_anns_all.extend(refcoco_anns)

        refcocop_anns = self.refcocop.get_annotations_and_captions(raw_img_id)
        if refcocop_anns is not None:
            ref_anns_all.extend(refcocop_anns)

        refcocog_anns = self.refcocog.get_annotations_and_captions(raw_img_id)
        if refcocog_anns is not None:
            ref_anns_all.extend(refcocog_anns)

        if len(ref_anns_all)==0:
            return None, None, None, None, None

        # merge caption
        captions = {}
        cap_idx = []
        all_boxes = []
        categories = []
        annotations = []

        for ann in ref_anns_all:
            id = ann['original_id']
            if captions.get(id) is None:
                captions[id] = {ann['caption']}
                cap_idx.append(id)
                all_boxes.append(ann['bbox'])
                categories.append(ann['category'])
                new_ann = {}
                new_ann['bbox'] = ann['bbox']
                new_ann['segmentation'] = ann['segmentation']
                annotations.append(new_ann)
            else:
                captions[id].add(ann['caption'])

        all_boxes = change_box(all_boxes, width, height)

        num_boxes = len(all_boxes)

        caption_gpt_format = caption_to_gpt_format(all_boxes, cap_idx, captions, categories)

        # generate final gpt question
        if type==0:
            gpt_format = generate_gpt_prompt(llava_desc, coco_gpt_format, caption_gpt_format, num_boxes, coco_cap_anns)
        elif type==1:
            gpt_format = generate_gpt_prompt_qa(llava_desc, coco_gpt_format, caption_gpt_format, num_boxes, coco_cap_anns)
        return gpt_format, annotations, num_boxes, height, width

        
if __name__ == "__main__":
    ask = GeneratePrompt()
    prompt = ask.load_data_and_generate_gpt_prompt(581857)
    print(prompt)