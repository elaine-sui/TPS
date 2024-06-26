#!/bin/bash

data_root='/path/to/root'
testsets=$1
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a
seed=$2

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 --seed $seed \
--tpt --ctx_init ${ctx_init} --img_aug --per_label --init_concepts \
--concept_type gpt4_no_cond \
--concat_concepts \
--logname test_tpt_with_concepts_simple_gpt4