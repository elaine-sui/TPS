#!/bin/bash

data_root='/path/to/data/bongard_hoi'
arch=RN50
bs=1
split=unseen_obj_unseen_act

python ./tpt_hoi.py ${data_root} --test_sets bongard \
-a ${arch} -b ${bs} --gpu 0 --tta_steps 64 \
--tpt --img_aug --bongard_split ${split} \
--logname test_tpt_hoi_${split}