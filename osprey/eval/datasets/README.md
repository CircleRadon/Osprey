# Data preparation for Open-Vocabulary Segmentation☕️
Dataset preparation follows [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md) and [Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md).

The datasets are assumed to exist in a directory specified by the environment variable `DETECTRON2_DATASETS`. Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  ADEChallengeData2016/
  cityscapes/
```
You can set the location for builtin datasets by export `DETECTRON2_DATASETS=/path/to/datasets`. The default is `./datasets` under the eval directory.

## ADE20k (A-150)
Dataset structure:
```
ADEChallengeData2016/
  images/
  annotations/
  objectInfo150.txt
  # download instance annotation
  annotations_instance/
  # generated by prepare_ade20k_sem_seg.py
  annotations_detectron2/
  # below are generated by prepare_ade20k_pan_seg.py
  ade20k_panoptic_{train,val}.json
  ade20k_panoptic_{train,val}/
  # below are generated by prepare_ade20k_ins_seg.py
  ade20k_instance_{train,val}.json
```
The directory annotations_detectron2 is generated by running `python datasets/prepare_ade20k_sem_seg.py`.

Download the instance annotation from http://sceneparsing.csail.mit.edu/:
```
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
```
Then, run `python datasets/prepare_ade20k_pan_seg.py`, to combine semantic and instance annotations for panoptic annotations.

Finally, run `python datasets/prepare_ade20k_ins_seg.py`, to extract instance annotations in COCO format.

## Cityscapes
Data structure:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note: to create labelTrainIds.png, first prepare the above structure, then run cityscapesescript with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

Note: to generate Cityscapes panoptic dataset, run cityscapesescript with:

```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```