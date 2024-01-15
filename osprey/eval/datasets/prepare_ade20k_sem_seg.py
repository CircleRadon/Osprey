"""
Reference: https://github.com/facebookresearch/Mask2Former/blob/main/datasets/prepare_ade20k_ins_seg.py
"""

import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image


def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    Image.fromarray(img).save(output)


if __name__ == "__main__":
    dataset_dir = (
        Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "ade" / "ADEChallengeData2016"
    )
    for name in ["training", "validation"]:
        annotation_dir = dataset_dir / "annotations" / name
        output_dir = dataset_dir / "annotations_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert(file, output_file)