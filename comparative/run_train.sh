#!/bin/bash

python train.py --tss 10000 --ssd '../SampleSplits_Age/' --nw 8 --bs 16 --sn 'age' --lr 0.01 

#python train.py --tss 10000 --ssd '../SampleSplits/' --nw 8 --bs 16 --sn 'label' --lr 0.01

#python train.py --tss 10000 --ssd '../SampleSplits_Sex/' --nw 8 --bs 16 --sn 'sex' --lr 0.01

