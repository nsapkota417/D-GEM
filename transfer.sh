#!/bin/bash
#$ -q gpu@@crc_gpu          #gpu@@csecri-v100 # gpu@@crc_gpu, gpu@@csecri
#$ -l gpu=1
#$ -M nsapkota@nd.edu
#$ -m abe
#$ -r y
#$ -o crclogs/$JOB_ID.out
#$ -e crclogs/$JOB_ID.err
#$ -pe smp 6

DST="/groups/dchen/nick/datasets"
mkdir -p "$DST"

for SRC in \
  "/users/nsapkota/VOS/data/datasets/cholecseg8k" \
  "/users/nsapkota/VOS/data/datasets/endovis" \
  "/users/nsapkota/VOS/data/datasets/sarrarp50"
do
  rsync -avh --progress "$SRC" "$DST/"
done
