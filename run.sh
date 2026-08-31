#!/bin/bash
#$ -q gpu@@crc_gpu          #gpu@@csecri-v100 # gpu@@crc_gpu, gpu@@csecri
#$ -l gpu=1
#$ -M nsapkota@nd.edu
#$ -m abe
#$ -r y
#$ -o crclogs/$JOB_ID.out
#$ -e crclogs/$JOB_ID.err
#$ -pe smp 6

source /users/nsapkota/afs/.bashrc 
conda activate pyt

# export CUDA_VISIBLE_DEVICES="${SGE_HGR_gpu_card// /,}"
# if [ -z ${SGE_HGR_gpu_card+x} ]; then 
#         SGE_HGR_gpu_card=-1

# fi

python src/train.py -cfg cfg/data/base.yaml \
  --task-type video \
  --train-csv /path/to/train.csv \
  --test-csv /path/to/test.csv \
  --use-memory


# python src/inference_crop_padding.py -cfg cfg/data/propagate_image.yaml


# For independent images, use: --task-type image --no-memory

conda deactivate
