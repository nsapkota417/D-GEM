#!/bin/bash 
#$ -M nsapkota@nd.edu
#$ -m abe
#$ -o crclogs/$JOB_NAME-$JOB_ID.log
#$ -j y
#$ -N CxT10FFo
#$ -q gpu@@csecri
#$ -l gpu_card=1

source /users/nsapkota/afs/.bashrc 
conda activate pyt

echo -e "Assigned GPU(s): ${SGE_HGR_gpu_card}\n"
echo -e "Starting Experiment =)"
echo -e "=-=-=-=-=-=-=-=-=-=-=-=-=\n"
cd /users/nsapkota/VOS

python src/train.py -cfg /users/nsapkota/VOS/exp/yamls/xmem_cholecseg8k_20260223_022253.yaml
