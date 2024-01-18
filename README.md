<p align="center" width="100%">
<img src="assets/osprey.png"  width="90%">
</p>

<div align=center>

![Static Badge](https://img.shields.io/badge/Osprey-v1-F7C97E) [![arXiv preprint](https://img.shields.io/badge/arxiv-2312.10032-ECA8A7?logo=arxiv)](https://arxiv.org/pdf/2312.10032.pdf) [![Dataset](https://img.shields.io/badge/Dataset-Hugging_Face-CFAFD4)](https://huggingface.co/datasets/AntGroup-MI/Osprey-724K) [![video](https://img.shields.io/badge/Watch_Video-36600E?logo=youtube&logoColor=green)](https://youtu.be/YsxqHBBnDfk) [![Static Badge](https://img.shields.io/badge/Try_Demo-6B88E3?logo=youtubegaming&logoColor=DAE4EE)](http://111.0.123.204:8000/) 
</div>


<div align=center>
Demo username & password: <b>osprey</b>
</div>

---

<div align=center>
<img src="./assets/qmsht.gif" />
<br>    
A part of <i>Along the River During the Qingming Festival</i> (清明上河图)
<br> 
<img src="./assets/qyqx.gif" />
<br>    
<i>Spirited Away</i> (千与千寻)
<br> 

</div>

## Updates 📌
[2024/1/15]🔥 We released the [evaluation](./osprey/eval/README.md) code.

[2023/12/29]🔥 We released the training code and [Osprey-724K](https://huggingface.co/datasets/AntGroup-MI/Osprey-724K) dataset.

[2023/12/18]🔥 We released the code, [osprey-7b model](https://huggingface.co/sunshine-lwt/Osprey-7b/tree/main) and [online demo](http://111.0.123.204:8000/) for Osprey.

## What is Osprey 👀
Osprey is a mask-text instruction tuning approach that extends MLLMs by incorporating pixel-wise mask regions into language instructions, enabling **fine-grained visual understanding**.  Based on input mask region, Osprey generate the semantic descriptions including **short description** and **detailed description**.

Our Osprey can seamlessly integrate with [SAM](https://github.com/facebookresearch/segment-anything) in point-prompt, box-prompt and segmentation everything modes to generate the semantics associated with specific parts or objects.

<img src="./assets/framework.png" width="800px">

## Watch Video Demo 🎥

<p align="center"> <a href="https://youtu.be/YsxqHBBnDfk"><img src="assets/video_cover.png" width="70%"></a> </p>


## Try Our Demo 🕹️
### Online demo
**Click** 👇 **to try our demo online.**

[**web demo**](http://111.0.123.204:8000/)

```
username: osprey
password: osprey
```

<table>
  <tr>
    <td style="text-align: center"><br>Point<br></td>
    <td><img src="./assets/demo_point.gif" width="700"></td>
  </tr>
    <tr>
    <td style="text-align: center"><br>Box<br></td>
    <td><img src="./assets/demo_box.gif" width="700"></td>
  </tr>
   </tr>
    <tr>
    <td style="text-align: center"><br>Everything<br></td>
    <td><img src="./assets/demo_all.gif" width="700"></td>
  </tr>
</table>

### Offline demo
💻 **requirments:** For this demo, it needs about `17GB` GPU memory for Osprey(15GB) and SAM(2GB).

1. First install [Gradio-Osprey-Demo](https://github.com/LiWentomng/gradio-osprey-demo).
2. Install Segment Anything.
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

3. Download all the checkpoints:

- [Osprey-7b](https://huggingface.co/sunshine-lwt/Osprey-7b/tree/main)
- [CLIP-convnext](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/blob/main/open_clip_pytorch_model.bin)
- [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

The default path of all the checkpoints:
```
├── demo
    ├── checkpoints
    │   ├── Osprey_7b
    │   └── sam_vit_b_01ec64.pth 
    └── open_clip_pytorch_model.bin
```

Or change the "mm_vision_tower" in `config.json`  of Osprey-7b model to the Absolute Path of `open_clip_pytorch_model.bin`.

4. Run `app.py`.
```
cd demo
python app.py --model checkpoints/Osprey_7b
```

## Install 🛠️
1. Clone this repository and navigate to Osprey folder
```
git clone https://github.com/CircleRadon/Osprey.git
cd Osprey
```
2. Install packages
```
conda create -n osprey python=3.10 -y
conda activate osprey
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Dataset 🌟
The all datasets for training can be found in [Dataset preparation](./dataset.md).

**Osprey-724K**: 🤗[Hugging Face](https://huggingface.co/datasets/AntGroup-MI/Osprey-724K)

`Osprey-724K` is an instruction dataset with mask-text pairs, containing around 724K GPT-generated multimodal dialogues to encourage MLLMs for fine-grained pixel-level image understanding. It contains object-level, part-level and additional instruction samples for robustness and flexibility.
<img src="./assets/data.png" />

## Training 🚀
- **Stage1: Image-Text Alignment Pre-training**
  - The pretrained projector weights for Convnext-large-CLIP can be found in [projector weights](https://huggingface.co/sunshine-lwt/osprey-v1.0-mlp2x-512px-convnext-pretrain-vicuna-7b-v1.5/tree/main).

- **Stage2: Mask-Text Alignment Pre-training**
  - Download [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main).
  - Download projector weights trained in stage1: [projector weights](https://huggingface.co/sunshine-lwt/osprey-v1.0-mlp2x-512px-convnext-pretrain-vicuna-7b-v1.5/tree/main).
  - Set `model_name_or_path` in `stage2.sh` to the path of `vicuna-7b-v1.5`.
  - Set `pretrain_mm_mlp_adapter` in `stage2.sh` to the path of `mm_projector`.
  - Set `vision_tower` in `stage2.sh` to the path of [Convnext-large-CLIP-model](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/blob/main/open_clip_pytorch_model.bin).
  - Run `sh scripts/stage2.sh`.

- **Stage3: End-to-End Fine-tuning**

  - Set `model_name_or_path` in `stage2.sh` to the path of `stage2 checkpoint`.
  - Set `vision_tower` in `stage2.sh` to the path of [Convnext-large-CLIP-model](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/blob/main/open_clip_pytorch_model.bin).
  - Run `sh scripts/stage3.sh`.


## Checkpoints 🤖

Osprey-7b model🤗: [model](https://huggingface.co/sunshine-lwt/Osprey-7b/tree/main)


<div align=center>
<img src="./assets/performance.png" />
</div>


## Evaluation 🔎
See [evaluation](./osprey/eval/README.md) for details.


## TODO List 📝
- [x] Release the checkpoints, inference codes and demo.
- [x] Release the dataset and training scripts.
- [x] Release the evaluation code.
- [ ] Release the code for data generation pipeline.


## Acknowledgement 💌
- [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA): the codebase we built upon.
- [SAM](https://github.com/facebookresearch/segment-anything): the demo uses the segmentation result from SAM as the input of Osprey.


## BibTeX 🖊️
```
@misc{Osprey,
  title={Osprey: Pixel Understanding with Visual Instruction Tuning},
  author={Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang and Jianke Zhu},
  year={2023},
  eprint={2312.10032},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
