#!/bin/bash

data_root='/path/to/root'
testsets=$1
arch=ViT-B/16
bs=64
lr=$2
seed=$3

python ./film_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 --seed $seed \
--img_aug --lr $lr --tta_steps 1 \
--text_shift \
--do_film \
--with_concepts --with_templates \
--concept_type gpt4 \
--logname test_film_text_shared_mean_init_template_and_gpt4_micro