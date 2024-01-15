"""
Reference: https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/eval_gpt_review.py
"""

import argparse
import json
import os

import openai
import time
from tqdm import tqdm
import requests

def get_eval(content: str, max_tokens: int):
    while True:
        try:
            messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }]
            ##########

            # change youre gpt interface here
            # ret = gpt_answer

            ##########
            break

        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(1)

    return ret


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('--question', help='path to question file')
    parser.add_argument('--context', help='path to gpt prompt file')
    parser.add_argument('--answer-list', nargs='+', default=[], help='gpt answer and model answer json files')
    parser.add_argument('--rule', help='gpt rule')
    parser.add_argument('--output', help='output json dir')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    f_q = json.load(open(os.path.expanduser(args.question)))
    f_ans1 = json.load(open(os.path.expanduser(args.answer_list[0])))
    f_ans2 = json.load(open(os.path.expanduser(args.answer_list[1])))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    os.makedirs('./result', exist_ok=True)

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    context_list = json.load(open(os.path.expanduser(args.context)))
    
    image_to_context = {context['image']: context for context in context_list}

    handles = []
    idx = 0
    
    for ques, ans1, ans2 in tqdm(zip(f_q, f_ans1, f_ans2)):

        inst = image_to_context[ques['image']]

        category = ques['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"category not found in rule file: {category}."
                    
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\{inst["prompt"]}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['question_id']),
            'category': category
        }
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens)
            print(review)
 
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            cur_js['answer1'] = ans1["text"]
            cur_js['answer2'] = ans2["text"]
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')

        idx += 1
        print(idx)
        
    review_file.close()