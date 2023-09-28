#!/bin/bash

GPU_IDX=1

python linear_probe.py --gpu_idx ${GPU_IDX} --train_first_conv --ft_strategy last-1
python linear_probe.py --gpu_idx ${GPU_IDX} --train_first_conv --ft_strategy linear_probe
python linear_probe.py --gpu_idx ${GPU_IDX} --train_first_conv --ft_strategy last-2
python linear_probe.py --gpu_idx ${GPU_IDX} --train_first_conv --ft_strategy last-3
python linear_probe.py --gpu_idx ${GPU_IDX} --train_first_conv --ft_strategy bn+last-3
python linear_probe.py --gpu_idx ${GPU_IDX} --train_first_conv --ft_strategy bn_only
python linear_probe.py --gpu_idx ${GPU_IDX} --train_first_conv --ft_strategy bn_nolinear
python linear_probe.py --gpu_idx ${GPU_IDX} --train_first_conv --ft_strategy full