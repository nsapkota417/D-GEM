#!/bin/bash
#$ -q gpu@@crc_gpu
#$ -l gpu=1
#$ -M nsapkota@nd.edu
#$ -m abe
#$ -r y
#$ -pe smp 6
#$ -cwd
#$ -o crclogs/$JOB_NAME.$JOB_ID.out
#$ -e crclogs/$JOB_NAME.$JOB_ID.err

set -eo pipefail


# ============================================================
# SHARED SETTINGS
# ============================================================

PROJECT_ROOT="/users/nsapkota/VOS"
CONFIG="${PROJECT_ROOT}/cfg/data/demo_inf.yaml"
INFERENCE_SCRIPT="${PROJECT_ROOT}/src/inference.py"

VIDEO="/groups/dchen/nick/demo/test_video_106.mp4"
RESULTS_ROOT="/groups/dchen/nick/demo/multiple_gpu_test_gto_1050p"

BATCH_SIZE=12
NUM_WORKERS=4


# ============================================================
# MODEL KEY | WEIGHTS
# ============================================================

MODELS=(
    "GTO|/users/nsapkota/VOS/results/demo_prep/MODEL_102_103_104_105_107_GTO/MODEL_102_103_104_105_107_GTO_ptT_ftT_bs8__20260731_180131/best_ep015_met7694_07312103.pth"

    # "PL|/users/nsapkota/VOS/results/demo_prep/MODEL_102_103_104_105_107_PL/MODEL_102_103_104_105_107_PL_ptT_ftT_bs8__20260731_180444/best_ep020_met7865_08030113.pth"

    # "GTO_AUG|/path/to/gto_aug_model.pth"
)


# ============================================================
# WORKER MODE
#
# Internally called by qsub as:
#
#   bash demo_run.sh run MODEL_KEY WEIGHTS
# ============================================================

if [[ "${1:-}" == "run" ]]; then
    MODEL_KEY="${2:?Missing MODEL_KEY}"
    WEIGHTS="${3:?Missing WEIGHTS path}"

    source /users/nsapkota/afs/.bashrc
    conda activate pyt

    cd "${PROJECT_ROOT}"

    mkdir -p "${PROJECT_ROOT}/crclogs"
    mkdir -p "${RESULTS_ROOT}"

    # --------------------------------------------------------
    # GPU setup
    #
    # Prefer CUDA_VISIBLE_DEVICES if CRC already set it.
    # Otherwise, fall back to the GPU assigned through SGE.
    # --------------------------------------------------------

    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && \
       [[ -n "${SGE_HGR_gpu_card:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="${SGE_HGR_gpu_card// /,}"
    fi

    echo "============================================================"
    echo "Inference job"
    echo "============================================================"
    echo "Model key            : ${MODEL_KEY}"
    echo "Weights              : ${WEIGHTS}"
    echo "Video                : ${VIDEO}"
    echo "Results              : ${RESULTS_ROOT}/${MODEL_KEY}"
    echo "Job ID               : ${JOB_ID:-unknown}"
    echo "Job name             : ${JOB_NAME:-unknown}"
    echo "Host                 : $(hostname)"
    echo "SGE_HGR_gpu_card     : ${SGE_HGR_gpu_card:-not set}"
    echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-not set}"
    echo "============================================================"

    if [[ ! -f "${WEIGHTS}" ]]; then
        echo "ERROR: checkpoint not found:"
        echo "  ${WEIGHTS}"
        exit 1
    fi

    if [[ ! -f "${CONFIG}" ]]; then
        echo "ERROR: config not found:"
        echo "  ${CONFIG}"
        exit 1
    fi

    if [[ ! -f "${INFERENCE_SCRIPT}" ]]; then
        echo "ERROR: inference script not found:"
        echo "  ${INFERENCE_SCRIPT}"
        exit 1
    fi

    if [[ ! -f "${VIDEO}" ]]; then
        echo "ERROR: input video not found:"
        echo "  ${VIDEO}"
        exit 1
    fi

    echo
    echo "nvidia-smi output:"
    nvidia-smi

    echo
    echo "PyTorch CUDA check:"

    python - <<'PY'
import os
import sys

import torch


print("Python executable       :", sys.executable)
print("PyTorch version         :", torch.__version__)
print("PyTorch CUDA build      :", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES    :", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available          :", torch.cuda.is_available())
print("CUDA device count       :", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Stopping instead of running inference on CPU."
    )

print("Current CUDA device     :", torch.cuda.current_device())
print("Visible GPU             :", torch.cuda.get_device_name(0))
print("GPU capability          :", torch.cuda.get_device_capability(0))
PY

    echo
    echo "Starting inference..."
    echo

    python "${INFERENCE_SCRIPT}" \
        --config "${CONFIG}" \
        --weights "${WEIGHTS}" \
        --video "${VIDEO}" \
        --results-dir "${RESULTS_ROOT}" \
        --run-name "${MODEL_KEY}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --gpu 0

    echo
    echo "Inference completed."
    echo "Model key : ${MODEL_KEY}"
    echo "Results   : ${RESULTS_ROOT}/${MODEL_KEY}"

    conda deactivate
    exit 0
fi


# ============================================================
# SUBMISSION MODE
#
# Run locally with:
#
#   bash demo_run.sh
#
# It submits one independent GPU job per model.
# ============================================================

cd "${PROJECT_ROOT}"

mkdir -p "${PROJECT_ROOT}/crclogs"
mkdir -p "${RESULTS_ROOT}"

for ENTRY in "${MODELS[@]}"; do
    IFS="|" read -r MODEL_KEY WEIGHTS <<< "${ENTRY}"

    if [[ ! -f "${WEIGHTS}" ]]; then
        echo "Skipping ${MODEL_KEY}: checkpoint not found"
        echo "  ${WEIGHTS}"
        echo
        continue
    fi

    echo "Submitting ${MODEL_KEY}..."
    echo "  Weights : ${WEIGHTS}"
    echo "  Results : ${RESULTS_ROOT}/${MODEL_KEY}"

    qsub \
        -N "demo_${MODEL_KEY}" \
        -q gpu@@crc_gpu \
        -l gpu=1 \
        -pe smp 6 \
        -cwd \
        -S /bin/bash \
        -o "${PROJECT_ROOT}/crclogs/demo_${MODEL_KEY}.\$JOB_ID.out" \
        -e "${PROJECT_ROOT}/crclogs/demo_${MODEL_KEY}.\$JOB_ID.err" \
        "$0" run "${MODEL_KEY}" "${WEIGHTS}"

    echo
done
