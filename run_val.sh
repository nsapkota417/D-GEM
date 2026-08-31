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

python src/train_image.py -cfg cfg/data/demo_val.yaml

conda deactivate

