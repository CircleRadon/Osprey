"""
Reference: https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/summarize_gpt_review.py
"""

import json
import os
from collections import defaultdict

import numpy as np

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-d', '--dir', default=None)
    parser.add_argument('-f', '--files', nargs='*', default=None)
    parser.add_argument('-i', '--ignore', nargs='*', default=None)
    parser.add_argument('-s', '--save', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    review_files = [x for x in os.listdir(args.dir) if x.endswith('.jsonl')]

    metrics = []
    for review_file in sorted(review_files):
        config = os.path.basename(review_file).replace('gpt4_text_', '').replace('.jsonl', '')
        scores = defaultdict(list)
        print(config)
        with open(os.path.join(args.dir, review_file) if args.dir is not None else review_file) as f:
            for review_str in f:
                review = json.loads(review_str)
                if args.ignore is not None and review['question_id'] in args.ignore:
                    continue
                if 'category' in review:
                    scores[review['category']].append(review['tuple'])
                    scores['all'].append(review['tuple'])
                else:
                    if 'tuple' in review:
                        scores['all'].append(review['tuple'])
                    else:
                        scores['all'].append(review['score'])
        summ_dict = defaultdict(list)
        for k, v in sorted(scores.items()):
            stats = np.asarray(v).mean(0).tolist()
            stats = [round(x, 3) for x in stats]
            # print(k, stats, round(stats[1]/stats[0]*100, 1))
            print(k, round(stats[1]/stats[0]*100, 2))
            summ_dict[k] = round(stats[1]/stats[0]*100, 2)
        print('=================================')
        metrics.append(summ_dict)
    
    if args.save:
        with open(os.path.join(args.dir, 'metric.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
            


        