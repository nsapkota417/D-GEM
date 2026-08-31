# Running D-GEM

This guide covers the two public workflows:

1. **Image segmentation:** train and infer on independent images.
2. **Video segmentation:** use D-GEM memory to train on sequences or propagate
   annotations from selected support frames to every frame of a video.

All commands below are run from the repository root after activating the
project environment:

```bash
conda activate pyt
```

The public entry points are:

```text
src/train.py    # training
src/infer.py    # checkpoint inference
```

## 1. Configure the project

The repository keeps one base configuration and one D-GEM model profile:

```text
cfg/data/base.yaml
cfg/model/dgem.yaml
```

Before a new experiment, update `cfg/data/base.yaml` to match the data:

- `data.num_class`: number of semantic classes, including background.
- `data.code_to_class`: map source mask values to contiguous class IDs.
- `data.mask_col`: the CSV column containing mask paths. For example, use
  `pl` when pseudo-label paths are stored in a `pl` column.
- `train.resize_h`, `train.resize_w`, `train.bs`, and `train.epochs`.
- `train.model_path` and `experiment.project_path` for output locations.

The model profile is merged automatically by both public commands. Memory is
controlled from the command line with `--use-memory` or `--no-memory` during
training. A checkpoint must use the same DINOv3 variant, class count, and image
resolution at inference; memory can be enabled later for video inference.

## Modular workflow: train images, then propagate video context

For an ordinary train/test split, start with **image-only training**. This
optimizes the DINOv3 encoder and segmentation decoder on independent labeled
rows and is often the simplest, most data-efficient baseline.

D-GEM memory is essentially training-free: its per-video transient and anchor
state is created from support masks and model predictions at inference time,
then discarded after that video. Video fine-tuning can optionally learn the
memory-fusion scalars, but it is not required. Consequently, an image-only
checkpoint can be used directly for video propagation as long as the DINOv3
variant, class count, and resize settings are unchanged:

```bash
# 1. Train the encoder/decoder with a conventional image train/test split.
python src/train.py -cfg cfg/data/base.yaml \
  --task-type image \
  --train-csv /path/to/train.csv \
  --test-csv /path/to/test.csv \
  --no-memory

# 2. Attach D-GEM memory at video inference using selected support annotations.
python src/infer.py -cfg cfg/data/base.yaml \
  --task-type video \
  --support-csv /path/to/supports.csv \
  --test-csv /path/to/test_frames.csv \
  --weights /path/to/image_checkpoint.pth \
  --save-preds
```

The inference runner recognizes image-only checkpoint keys and loads them into
the video wrapper's encoder/decoder automatically. Memory-specific parameters
that are absent from an image checkpoint retain the values in
`cfg/model/dgem.yaml`.

### Run D-GEM without memory

Memory is optional. `--no-memory` runs the same DINOv3 encoder/decoder as an
independent-frame segmentation model, which is a strong and useful baseline
for conventional train/test evaluation. Use it when sequential context is not
needed, then switch to `--use-memory` when annotated support frames and
temporal propagation are beneficial.

```bash
python src/train.py -cfg cfg/data/base.yaml \
  --task-type video \
  --train-csv /path/to/train.csv \
  --test-csv /path/to/test.csv \
  --no-memory
```

## 2. CSV manifests

### Common columns

Every manifest needs an image-path column named `img`. The configured mask
column is normally `mask`, but can be changed in `cfg/data/base.yaml`.

```text
img,mask
/path/to/image_0001.png,/path/to/mask_0001.png
```

Use a blank value or `-` in the mask column when ground truth is unavailable:

```text
img,mask
/path/to/image_0001.png,-
```

### Video columns

Video manifests need `video_src` in addition to `img` and the configured mask
column. `video_clip` is optional; when absent, D-GEM treats each `video_src`
as one clip. `frame_idx` is recommended because it preserves the original
temporal position for memory and output ordering.

```text
img,mask,video_src,video_clip,frame_idx
/data/v01/frame000000.png,/data/v01/mask000000.png,v01,clip_01,0
/data/v01/frame000001.png,-,v01,clip_01,1
```

## 3. Train/test split training

Use this workflow when train and test are distinct dataset partitions.

### Image mode

```bash
python src/train.py -cfg cfg/data/base.yaml \
  --task-type image \
  --train-csv /path/to/train.csv \
  --test-csv /path/to/test.csv \
  --no-memory
```

Image mode treats each row independently. Extra video metadata columns are
allowed but are not needed.

### Video mode

```bash
python src/train.py -cfg cfg/data/base.yaml \
  --task-type video \
  --train-csv /path/to/train.csv \
  --test-csv /path/to/test.csv \
  --use-memory
```

For sparse video training, the training CSV may contain unlabeled frames. Set:

```yaml
train:
  use_stream: true
  rollout_mode: full
```

The video workflow advances memory through all frames in the rollout, while
loss is computed only for annotated query frames. Each training clip needs at
least two annotated frames: one is selected as a support frame and another is
used as a supervised query frame.

## 4. Propagation from selected annotated frames

Propagation is video inference in which the **support CSV controls exactly
which annotations initialize D-GEM memory**.

Prepare two manifests for the same video IDs:

- `supports.csv`: contains only the few frames selected for annotation. Every
  support row must have a valid mask path.
- `test_frames.csv`: contains every frame that should receive a prediction.
  Its mask paths are optional; use `-` for unannotated frames.

The support and test rows are paired by `video_src` and, when present,
`video_clip`. Each test video must have at least one labeled support row with
the same identifiers in `supports.csv`.

Example:

```text
# supports.csv
img,mask,video_src,video_clip,frame_idx
/data/v01/frame000000.png,/data/v01/mask000000.png,v01,clip_01,0
/data/v01/frame000050.png,/data/v01/mask000050.png,v01,clip_01,50

# test_frames.csv
img,mask,video_src,video_clip,frame_idx
/data/v01/frame000000.png,/data/v01/mask000000.png,v01,clip_01,0
/data/v01/frame000001.png,-,v01,clip_01,1
/data/v01/frame000002.png,-,v01,clip_01,2
```

Run inference with the trained checkpoint:

```bash
python src/infer.py -cfg cfg/data/base.yaml \
  --task-type video \
  --support-csv /path/to/supports.csv \
  --test-csv /path/to/test_frames.csv \
  --weights /path/to/checkpoint.pth \
  --output-dir outputs/v01_propagation \
  --save-preds
```

The runner initializes anchor memory from the labeled `supports.csv` rows, then
processes every row of `test_frames.csv` in temporal order. It never uses a
test mask as a support label. If a test row has a valid mask, that mask is used
only to calculate metrics.

## 5. Checkpoint inference on a test set

### Video checkpoint inference

Use a support manifest and a test manifest just as in propagation. This is also
the correct command for a video test set: each test video must have matching
annotated support frames in the support CSV.

```bash
python src/infer.py -cfg cfg/data/base.yaml \
  --task-type video \
  --support-csv /path/to/test_supports.csv \
  --test-csv /path/to/test.csv \
  --weights /path/to/checkpoint.pth \
  --output-dir outputs/test_video \
  --save-preds
```

If `test.csv` contains ground-truth mask paths, D-GEM reports mean per-frame
mIoU. If masks are blank or `-`, it still produces predictions but reports no
evaluation score for those frames.

### Image checkpoint inference

Image inference does not use a support CSV:

```bash
python src/infer.py -cfg cfg/data/base.yaml \
  --task-type image \
  --test-csv /path/to/test.csv \
  --weights /path/to/checkpoint.pth \
  --output-dir outputs/test_image \
  --save-preds
```

Omit `--save-preds` when only evaluation numbers are needed.

## 6. Inference outputs

Every inference run writes:

```text
<output-dir>/report.csv
<output-dir>/summary.json
```

`report.csv` contains one row per processed test frame, the optional prediction
path, and an `miou` value when that frame has a ground-truth mask. `summary.json`
records processed-frame count, evaluated-frame count, mean frame mIoU, and
whether PNG predictions were requested.

With `--save-preds`, label-map PNGs are also written below:

```text
<output-dir>/predictions/<video_src>/<video_clip>/<frame-name>.png
```

## 7. Useful command-line overrides

`src/train.py` overrides base-config values with:

```text
--task-type image|video
--data-csv PATH             # use one manifest for both train and test
--train-csv PATH
--test-csv PATH
--use-memory
--no-memory
```

`src/infer.py` accepts:

```text
--task-type image|video
--support-csv PATH          # required for video inference
--test-csv PATH
--weights PATH
--output-dir PATH
--save-preds
--gpu INDEX
```

## 8. Server workflow

Update the server checkout before running:

```bash
cd /path/to/D-GEM
git pull origin main
conda activate pyt
```

If Git reports a local modified file, inspect it before pulling:

```bash
git diff
git stash push -m "temporary server edit"
git pull origin main
```
