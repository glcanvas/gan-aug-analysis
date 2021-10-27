#!/bin/bash

isic19_root="/mnt/tank/scratch/nduginets"

GPU=$1

val_csv="/nfs/home/nduginets/gan-aug-analysis/splits/isic2019-val.csv"

train_plain_old_clf="nfs/home/nduginets/gan-aug-analysis/splits/percentages/1_0"
train_pix2pix_clf="1488"

SPLITS=9

for split in $(seq 0 $SPLITS); do
CUDA_VISIBLE_DEVICES=$GPU python3 train_comet_csv.py with \
				train_root=${isic19_root} train_csv=${train_plain_old_clf}/train_${split}.csv epochs=100\
				val_root=${isic19_root} val_csv=${val_csv} model_name="inceptionv4" exp_desc="Real" exp_name="gans.train_Real.inceptionv4.split${split}"
done