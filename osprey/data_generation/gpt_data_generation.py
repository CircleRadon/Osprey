from ask_gpt import askGPT
from generate_gpt_prompt import GeneratePrompt, COCODataset
import json
from tqdm import tqdm
import re
import argparse

QUESTIONS =  [
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

class COCODataConvert():
    def __init__(self):
        self.gpt = askGPT()
        self.generate_gpt_prompt = GeneratePrompt()
        self.coco = COCODataset()
    
    def generate_conversation(self, output_file):
        results = []
        imgs = self.coco.img_ids
        sum = 0
        for id in (tqdm(imgs)):
            result = {}
            prompt, annotations, num_boxes, height, width = self.generate_gpt_prompt.load_data_and_generate_gpt_prompt_description(id,1)
            # print(prompt)
            if prompt == None:
                # print("none")
                continue

            while True:
                try:
                    ret = self.gpt.ask_gpt_conversation(prompt)
                    description =ret.json()['data']['reply']

                    conversations = []
                    description_list = description.split('\n\n')
                    for i, des in enumerate(description_list):
                        conv = {}
                        if i%2==0:
                            conv['from'] = "human"
                            conv['value'] = re.findall(r"Question.*:\ (.*)",des)[0]
                        else:
                            conv['from'] = "gpt"
                            conv['value'] = re.findall(r"Answer.*:\ (.*)",des)[0]
                        conversations.append(conv)
                    break
                except:
                    print(ret)
            

            img_info = self.coco.coco.load_imgs([id])[0]
            result['id'] = id
            result['file_name'] = img_info['file_name']
            result['conversations'] = conversations
            result['annotation'] = annotations
            result['height'] = height
            result['width'] = width
            results.append(result)

            sum+=1
            print("num:", sum)

            if sum%100==0:
                f = json.dumps(results)
                f2 = open(output_file, 'w')
                f2.write(f)
                f2.close()

        f = json.dumps(results)
        f2 = open(output_file, 'w')
        f2.write(f)
        f2.close()


    def generate_short_conversation(self, output_file):
        results = []
        imgs = self.coco.img_ids
        sum = 0

        for id in (tqdm(imgs)):
            # print(id)
            result = {}
            prompt, annotations, num_boxes, height, width = self.generate_gpt_prompt.load_data_and_generate_gpt_prompt_description(id,1)
            # print(prompt)
            if prompt == None:
                # print("none")
                continue
            while True:
                try:
                    ret = self.gpt.ask_gpt_short_conversation(prompt)
                    description =ret.json()['data']['reply']

                    conversations = []
                    description_list = description.split('\n\n')

                    for i, des in enumerate(description_list):
                        conv = {}
                        if i%2==0:
                            conv['from'] = "human"
                            conv['value'] = re.findall(r"Question.*:\ (.*)",des)[0]
                        else:
                            conv['from'] = "gpt"
                            conv['value'] = re.findall(r"Answer.*:\ (.*)",des)[0]
                        conversations.append(conv)
                    break
                except:
                    print(ret)                
            
            img_info = self.coco.coco.load_imgs([id])[0]
            result['id'] = id
            result['file_name'] = img_info['file_name']
            result['conversations'] = conversations
            result['annotation'] = annotations
            result['height'] = height
            result['width'] = width
            results.append(result)

            sum+=1
            print("num:",sum)

            if sum%100==0:
                f = json.dumps(results)
                f2 = open(output_file, 'w')
                f2.write(f)
                f2.close()

        f = json.dumps(results)
        f2 = open(output_file, 'w')
        f2.write(f)
        f2.close()


    def generate_descriptions(self, output_file):
        results = []
        imgs = self.coco.img_ids
        sum = 0
        for id in tqdm(imgs):
            result = {}
            prompt, annotations, num_boxes, height, width = self.generate_gpt_prompt.load_data_and_generate_gpt_prompt_description(id)
            # print(prompt)

            if prompt == None:
                # print("None")
                continue

            while True:
                try:
                    description = self.gpt.ask_gpt(prompt)

                    description_list = description.split('\n\n')
                    break
                except:
                    print(description)


            img_info = self.coco.coco.load_imgs([id])[0]
            result['id'] = id
            result['file_name'] = img_info['file_name']
            result['description'] = description_list
            result['annotation'] = annotations
            result['height'] = height
            result['width'] = width
            results.append(result)

            sum+=1
            print("num:",sum)

            if sum%100==0:
                f = json.dumps(results)
                f2 = open(output_file, 'w')
                f2.write(f)
                f2.close()

        f = json.dumps(results)
        f2 = open(output_file, 'w')
        f2.write(f)
        f2.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data generation pipline', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--type', help='generate data type', default='description')
    parser.add_argument('--outputfile', help='output file name', default='description_gpt4_data.json')
    args = parser.parse_args()

    convert = COCODataConvert()

    if args.type=='description':
        convert.generate_descriptions(args.output_file)
    elif args.type=='conversation':
        convert.generate_conversation(args.output_file)
    elif args.type=='short-form':
        convert.generate_short_conversation(args.output_file)
    else:
        raise NotImplementedError
