# Dataset Preparation

- Osprey-724K 🤗 [download](https://huggingface.co/datasets/AntGroup-MI/Osprey-724K)

| Data | Size |
| --- | ---: |
| osprey_short_form.json | 57 MB |
| osprey_conversation.json |  106 MB |
| osprey_detail_description.json | 63.4 MB |
| osprey_part_level.json | 153 MB |
| osprey_lvis_positive_negative.json | 140 MB |


- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip), `imgs` should contain all the images including training set and validation set.
- pascal_part: [train.json](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/pascalpart_train.json?download=true), [VOCdevkit](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar).
- partImagenet: [train_format.json](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/partImagenet_train_format.json?download=true),
[PartImageNet_OOD](https://drive.google.com/file/d/19kA8-pAxssQI0GD5H8y8KESGaALwChtx/view?usp=sharing).
- refcocos: [refcoco](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/finetune_refcoco_train_with_mask.json?download=true), [refcoco+](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/finetune_refcoco%2B_train_with_mask.json?download=true).
- vg: [vg_train_with_mask.json](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/vg_train_with_mask.json?download=true) (mask is generated from [HQ-SAM](https://github.com/SysCV/sam-hq)), images can be downloaded from [OpendataLab](https://opendatalab.com/OpenDataLab/Visual_Genome_Dataset_V1_dot_2), `image` should contain all the vg images(VG_100K and VG_100K_2).
- vcr: [vcr](https://visualcommonsense.com/download/).

After downloading all of them, organize the data as follows in `./data`,


```
├── coco
│   ├── annotations
│   │   └── instances_train2017.json
│   └── imgs
├── part data
│   ├── pascal_part
│   │   ├── train.json
│   │   └── VOCdevkit
│   └── partImagenet
│       ├── train_format.json
│       └── train
├── refcocos
│   ├── finetune_refcoco_train_with_mask.json
│   └── finetune_refcoco+_train_with_mask.json
├── Osprey-724K
│   ├── osprey_short_form.json
│   ├── osprey_conversation.json
│   ├── osprey_detail_description.json
│   ├── osprey_part_level.json
│   └── osprey_lvis_positive_negative.json
├── vg
│   ├── vg_train_with_mask.json
│   └── image
└── vcr
    ├── train.jsonl
    └── vcr1images
```
