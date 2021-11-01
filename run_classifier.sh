#!/bin/bash
  
isic19_root="/mnt/tank/scratch/nduginets"


val_csv="/nfs/home/nduginets/gan-aug-analysis/splits/isic2019-val.csv"

train_plain_old_clf="/nfs/home/nduginets/gan-aug-analysis/splits/percentages/1_0"

DEVICES=$1
SPLITS=$(echo $2 | tr ";" "\n")

for split in $SPLITS; do
CUDA_VISIBLE_DEVICES=$DEVICES python3 train_comet_csv.py with \
                                train_root=${isic19_root} train_csv=${train_plain_old_clf}/train_${split}.csv epochs=100\
                                val_root=${isic19_root} val_csv=${val_csv} model_name="inceptionv4" exp_desc="Real"\
                                exp_name="gans.train_Real.inceptionv4.split${split}"
done

# CUDA_VISIBLE_DEVICES=0 python3 train_comet_csv.py with train_root="/mnt/tank/scratch/nduginets" train_csv="/nfs/home/nduginets/gan-aug-analysis/splits/percentages/1_0/train_0.csv" epochs=100 val_root="/mnt/tank/scratch/nduginets" val_csv="/nfs/home/nduginets/gan-aug-analysis/splits/isic2019-val.csv" model_name="inceptionv4" exp_desc="Real" exp_name="gans.train_Real.inceptionv4.split0"