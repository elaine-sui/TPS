#!/bin/bash

data_root='/path/to/data/bongard_hoi'
arch=RN50
bs=1
split=seen_obj_unseen_act

python ./shift_hoi.py ${data_root} --test_sets bongard \
-a ${arch} -b ${bs} --gpu 0 --tta_steps 64 --seed $1 \
--text_shift --do_shift --img_aug --bongard_split ${split} \
--per_label \
--logname test_shift_per_label_hoi_${split}