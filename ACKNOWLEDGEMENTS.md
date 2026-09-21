# Third-party acknowledgements

D-GEM is released under the repository's [MIT License](LICENSE). Third-party
software and model materials remain subject to their own licenses; the D-GEM
license does not replace or modify those terms.

## Runtime dependency

### DINOv3

D-GEM uses a pretrained **DINOv3** visual encoder developed by Meta AI
Research. The implementation loads Meta's `facebook/dinov3-*` checkpoints
through the Hugging Face Transformers API; DINOv3 source code and pretrained
weights are not vendored in this repository.

- Project: <https://github.com/facebookresearch/dinov3>
- Paper: <https://arxiv.org/abs/2508.10104>
- License: <https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md>

DINOv3 code and model weights are governed by the DINOv3 License. That license
requires research publications using the DINO materials to acknowledge their
use. Users downloading, fine-tuning, or redistributing DINOv3 weights must
review and comply with those terms. In particular, a released checkpoint that
contains DINOv3 parameters may constitute redistribution of DINO materials and
should be accompanied by the applicable DINOv3 License.

We thank the DINOv3 authors and Meta AI Research for making the models
available to the research community.

## Referenced and evaluated systems

The following systems are discussed or evaluated as prior work and baselines
in the D-GEM paper. Their source code and weights are not bundled with this
repository, and they are not required by the supported D-GEM runtime.

### DINOv2

DINOv2 is acknowledged as foundational prior work in self-supervised visual
representation learning. The current public D-GEM implementation uses DINOv3,
not DINOv2.

- Project: <https://github.com/facebookresearch/dinov2>
- Paper: <https://arxiv.org/abs/2304.07193>
- License: <https://github.com/facebookresearch/dinov2/blob/main/LICENSE>

The main DINOv2 repository identifies its code and model weights as released
under the Apache License 2.0. Users should also inspect any additional licenses
associated with particular upstream artifacts they obtain.

### SAM 2

SAM 2 is evaluated as a video-segmentation baseline in the D-GEM paper. The
supported D-GEM release does not redistribute the SAM 2 implementation or
weights.

- Project: <https://github.com/facebookresearch/sam2>
- Paper: <https://arxiv.org/abs/2408.00714>
- License: <https://github.com/facebookresearch/sam2/blob/main/LICENSE>

SAM 2 is released under the Apache License 2.0. We thank the SAM 2 authors and
Meta AI Research for releasing the model and evaluation resources.

### SAM 3

SAM 3 is evaluated as a segmentation baseline in the D-GEM paper. The
supported D-GEM release does not redistribute the SAM 3 implementation or
weights.

- Project: <https://github.com/facebookresearch/sam3>
- License: <https://github.com/facebookresearch/sam3/blob/main/LICENSE>

SAM 3 materials are governed by the SAM License. Its terms require research
publications using the SAM materials to acknowledge their use and require the
license to accompany redistribution of SAM materials or derivatives. Users who
obtain or redistribute SAM 3 materials must review and comply with that
license.

## No endorsement

DINO, DINOv2, DINOv3, Segment Anything, SAM 2, SAM 3, Meta, and related names
may be trademarks of their respective owners. These acknowledgements describe
technical dependencies and research comparisons and do not imply endorsement
of D-GEM by Meta or any upstream author.
