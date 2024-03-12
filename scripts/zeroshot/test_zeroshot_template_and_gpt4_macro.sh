#!/bin/bash

data_root='/path/to/root'
testsets=$1
arch=ViT-B/16
bs=1

python ./shift_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tta_steps 0 --with_concepts --with_templates \
--concept_type gpt4 \
--macro_pooling \
--logname test_zeroshot_template_and_gpt4_macro