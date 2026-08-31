# D-GEM
Sparsely Supervised Surgical Video Segmentation with Reliable Asymmetric Dual Memory

Abstract. Surgical video segmentation often operates on long image sequences for which dense annotation is prohibitively expensive, making sparse supervision more practical. Existing approaches suffer from temporal drift due to error accumulation, while formulations based on video object segmentation (VOS) or independent per-frame processing rely on restrictive assumptions or dense labels. Recent VOS foundation models offer strong visual representations, which are often per-object and tightly coupled with memory, limiting scalability and flexibility under domain shift in surgical videos. We propose a reliability-gated asymmetric dual-memory framework built on self-supervised visual representations of DINOv3. Our model accounts for both short-term temporal continuity and long-term semantic stability using a gated transient memory with bounded capacity and an evolving anchor memory that incrementally builds semantic representations without requiring complete first-frame class coverage. By decoupling memory from the encoder, our framework enables direct use of pretrained vision foundation encoders and facilitates data-efficient adaptation under surgical video domain shift. Experiments on multiple surgical video datasets demonstrate improved temporal consistency and robustness compared to state-of-the-art baselines.

Accepted for publication in the 29th INTERNATIONAL CONFERENCE ON MEDICAL IMAGE COMPUTING AND COMPUTER ASSISTED INTERVENTION (MICCAI 2026).
Codes will be released soon. 

## Data modes

D-GEM has two canonical dataset implementations:

- `src/dataset_image.py` for independent image training and evaluation.
- `src/dataset_video.py` for sequential D-GEM training and propagation.

### Sparse video manifests

Video datasets are CSV files with at least these columns:

```text
img,mask,video_src,video_clip
/data/video_01/frame000001.png,/data/video_01/masks/frame000001.png,video_01,clip_01
/data/video_01/frame000002.png,-,video_01,clip_01
```

The `mask` value may be blank or `-` for an unannotated frame. `dataset_video.py`
uses only annotated frames as support or supervised query frames while preserving
all image paths for sequential memory rollouts. A training clip requires at least
two annotated frames: one support frame and one supervised query frame.

For sparse temporal training, set `train.use_stream: true` and use a rollout that
contains your annotations (normally `train.rollout_mode: full`). This lets D-GEM
advance its transient and anchor memories through unlabeled frames while applying
loss only on the annotated query frames.
