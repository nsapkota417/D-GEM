# D-GEM: Sparsely Supervised Surgical Video Segmentation with Reliable Asymmetric Dual Memory

> D-GEM combines a DINOv3 segmentation backbone with reliability-gated
> transient memory and evolving semantic anchors for long, sparsely annotated
> surgical videos.

**Nishchal Sapkota, Yejia Zhang, Bofang Zheng, Xianshi Ma, Haoyan Shi,
Yohannes Mariam, and Danny Z. Chen**<br>
Department of Computer Science and Engineering, University of Notre Dame

Accepted at the 29th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2026).

<p align="center">
  <a href="https://conferences.miccai.org/2026/en/PROMOTIONAL-KIT.html">
    <img src="assets/miccai2026-logo.png" alt="MICCAI 2026 official logo" width="260">
  </a>
</p>

<sub>Official MICCAI 2026 artwork from the <a href="https://conferences.miccai.org/2026/en/PROMOTIONAL-KIT.html">MICCAI promotional kit</a>.</sub>

| Paper | Code | Practical guide |
| --- | --- | --- |
| Sparsely Supervised Surgical Video Segmentation with Reliable Asymmetric Dual Memory | This repository | [Training and inference guide](docs/RUNNING.md) |

## Contents

- [Overview](#overview)
- [Why D-GEM](#why-d-gem)
- [Results from the camera-ready paper](#results-from-the-camera-ready-paper)
- [Quick start](#quick-start)
- [Data manifests](#data-manifests)
- [Training](#training)
- [Inference and propagation](#inference-and-propagation)
- [Repository layout](#repository-layout)
- [Citation](#citation)

## Overview

Dense surgical-video annotations are expensive, yet long videos demand temporal
reasoning beyond independent frame segmentation. D-GEM is designed for sparse
frame-level supervision: annotated frames provide semantic support, and the
model processes the full video sequentially to predict a segmentation for every
frame.

The camera-ready paper introduces an asymmetric two-bank memory design:

| Component | Role | Update behavior |
| --- | --- | --- |
| **DINOv3 encoder + segmentation head** | Produces per-frame patch features and semantic logits | Pretrained visual representation with lightweight adaptation |
| **Gated Transient Memory (GTM)** | Preserves recent temporal context | Writes only reliable predictions under a bounded sliding-window budget |
| **Evolving Anchor Memory (EAM)** | Preserves long-term class-specific semantics | Initializes from annotations and incrementally refines diverse, reliable anchors |
| **Dual-memory attention** | Retrieves and fuses temporal and semantic context | Learnable asymmetric fusion of GTM and EAM context |

This decouples memory from the encoder, allowing the same DINOv3 representation
to be used for image-only segmentation, sparse-video training, and annotation
propagation.

## Why D-GEM

| Challenge in surgical video | D-GEM response |
| --- | --- |
| Long temporal gaps between annotations | GTM carries recent reliable context across unannotated frames |
| Occlusion, smoke, and abrupt appearance changes | Reliability gating limits error accumulation in transient memory |
| Classes missing from the first annotated frame | EAM can add or refine class anchors from later annotated support frames |
| Domain shift with limited labels | DINOv3 features and memory are decoupled for data-efficient adaptation |
| Per-object VOS assumptions do not fit semantic segmentation | D-GEM makes dense, multi-class semantic predictions directly |

## Results from the camera-ready paper

Average mIoU (%) on three surgical-video datasets. D-GEM uses 34.1M parameters
(5.4M trainable in the frozen-encoder setting).

### One labeled support frame

| Method | CholecSeg8k | EndoVis 2018 | SAR-RARP50 | Average |
| --- | ---: | ---: | ---: | ---: |
| SAM 2 | 72.4 | 50.1 | 33.6 | 52.0 |
| SAM 3 | 76.7 | 54.8 | 36.3 | 55.9 |
| XMem | 70.3 | 45.8 | 41.2 | 52.4 |
| DINOv3Seg | 75.4 | 53.1 | 50.9 | 59.8 |
| DINOv3Seg-V | 76.0 | 53.6 | 51.1 | 60.2 |
| **D-GEM** | **78.9** | **56.3** | **52.5** | **62.6** |

### Adaptation with 10 labeled frames

| Method | CholecSeg8k | EndoVis 2018 | SAR-RARP50 | Average |
| --- | ---: | ---: | ---: | ---: |
| SAM 2 [R] | 76.3 | 52.7 | 44.6 | 57.9 |
| SAM 3 [R] | 81.1 | 56.4 | 54.3 | 63.9 |
| XMem | 74.6 | 50.2 | 58.6 | 61.1 |
| DINOv3Seg | 85.4 | 57.2 | 62.0 | 68.2 |
| DINOv3Seg-V | 85.6 | 59.6 | 67.7 | 71.0 |
| **D-GEM** | **86.5** | **64.4** | **71.6** | **74.2** |

The paper also reports lower early-to-late temporal drift for D-GEM, especially
on long SAR-RARP50 videos, and shows complementary gains from the transient
and evolving-anchor memory components.

## Quick start

```bash
git clone <repository-url>
cd D-GEM
conda activate pyt
```

Configure label handling and training defaults in `cfg/data/base.yaml`, then
choose one of the public commands below.

| Goal | Command |
| --- | --- |
| Train image segmentation | `python src/train.py -cfg cfg/data/base.yaml --task-type image --train-csv TRAIN.csv --test-csv TEST.csv --no-memory` |
| Train video segmentation | `python src/train.py -cfg cfg/data/base.yaml --task-type video --train-csv TRAIN.csv --test-csv TEST.csv --use-memory` |
| Infer on independent images | `python src/infer.py -cfg cfg/data/base.yaml --task-type image --test-csv TEST.csv --weights CHECKPOINT.pth --save-preds` |
| Propagate selected video annotations | `python src/infer.py -cfg cfg/data/base.yaml --task-type video --support-csv SUPPORTS.csv --test-csv FRAMES.csv --weights CHECKPOINT.pth --save-preds` |

See [docs/RUNNING.md](docs/RUNNING.md) for complete commands, output locations,
server setup, and sparse-label behavior.

## Data manifests

The configured mask column defaults to `mask`; change `data.mask_col` in
`cfg/data/base.yaml` if labels live in another column such as `pl`.

| Mode | Required columns | Notes |
| --- | --- | --- |
| Image | `img`, configured mask column | A mask may be `-` at inference time |
| Video | `img`, `video_src`, configured mask column | `video_clip` is optional; `frame_idx` is recommended for temporal ordering |
| Video support manifest | Same as video, with valid masks on support rows | Defines exactly which annotated frames initialize memory during propagation |

Example sparse-video manifest:

```text
img,mask,video_src,video_clip,frame_idx
/data/v01/frame000000.png,/data/v01/mask000000.png,v01,clip_01,0
/data/v01/frame000001.png,-,v01,clip_01,1
```

## Training

### Train/test split

```bash
python src/train.py -cfg cfg/data/base.yaml \
  --task-type video \
  --train-csv /path/to/train.csv \
  --test-csv /path/to/test.csv \
  --use-memory
```

For sparse sequential training, retain `train.use_stream: true` and
`train.rollout_mode: full`. Every frame advances memory, while supervision is
applied only on frames with a valid mask. Each train clip requires at least two
annotated frames: one support and one supervised query.

### Image-only mode

```bash
python src/train.py -cfg cfg/data/base.yaml \
  --task-type image \
  --train-csv /path/to/train.csv \
  --test-csv /path/to/test.csv \
  --no-memory
```

## Inference and propagation

Video propagation uses **two CSV files**:

1. `SUPPORTS.csv` contains only the annotated frames that should initialize
   D-GEM memory. It controls the support frames explicitly.
2. `FRAMES.csv` contains every test frame to process. Its masks are optional;
   labels, when present, are used only for evaluation and never as supports.

```bash
python src/infer.py -cfg cfg/data/base.yaml \
  --task-type video \
  --support-csv /path/to/SUPPORTS.csv \
  --test-csv /path/to/FRAMES.csv \
  --weights /path/to/checkpoint.pth \
  --output-dir outputs/propagation \
  --save-preds
```

Support and test frames are matched by `video_src` and `video_clip`. Each test
video therefore needs at least one labeled matching support frame. The command
always writes `report.csv` and `summary.json`; `--save-preds` additionally
writes PNG label maps under the output directory.

## Repository layout

```text
cfg/data/base.yaml       Shared data and experiment defaults
cfg/model/dgem.yaml      D-GEM memory profile
docs/RUNNING.md          Detailed training and inference guide
src/train.py             Unified image/video training entry point
src/infer.py             Unified image/video inference entry point
src/dataset_image.py     Independent-image dataset
src/dataset_video.py     Sequential sparse-video dataset
src/workflows/           Internal image and video training workflows
```

## Citation

```bibtex
@inproceedings{sapkota2026dgem,
  title={Sparsely Supervised Surgical Video Segmentation with Reliable Asymmetric Dual Memory},
  author={Sapkota, Nishchal and Zhang, Yejia and Zheng, Bofang and Ma, Xianshi and Shi, Haoyan and Mariam, Yohannes and Chen, Danny Z.},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2026}
}
```

## Acknowledgments

This work was supported in part by the Advanced Research Projects Agency for
Health (ARPA-H), Award 1AY2AX000049.
