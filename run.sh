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

# python src/train_image.py -cfg cfg/data/cisvismm_ureter_1min.yaml
# python src/train_image.py -cfg cfg/data/cisvismm_bile_10min.yaml
# python src/train_image.py -cfg cfg/data/cisvismm_thoraic_4min.yaml
python src/train_image.py -cfg cfg/data/demo_prep.yaml


# python src/inference.py -cfg /users/nsapkota/VOS/cfg/data/demo_inf.yaml


# python src/train_image.py -cfg cfg/data/cnh_pe.yaml
# python src/train_video.py -cfg cfg/data/cholecseg8k.yaml
# python src/train_video.py -cfg cfg/data/sarrarp50.yaml
# python src/train_video.py -cfg cfg/data/endovis.yaml

conda deactivate

