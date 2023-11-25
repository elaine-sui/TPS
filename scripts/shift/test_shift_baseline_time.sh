#!/bin/bash

data_root='data'
testsets=$1
arch=ViT-B/16
bs=64
lr=5e-3
seed=1234

python ./shift_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 --seed $seed \
--img_aug --lr $lr --tta_steps 1 \
--text_shift \
--do_shift \
--per_label \
--with_concepts \
--concept_type gpt4 \
--logname test_shift_text_per_class_mean_init_gpt4_runtime