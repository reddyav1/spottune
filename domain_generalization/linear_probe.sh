#!/bin/bash

GPU_IDX=0

python linear_probe.py --gpu_idx ${GPU_IDX} --ft_strategy last-1
python linear_probe.py --gpu_idx ${GPU_IDX} --ft_strategy linear_probe
python linear_probe.py --gpu_idx ${GPU_IDX} --ft_strategy last-2
python linear_probe.py --gpu_idx ${GPU_IDX} --ft_strategy last-3
python linear_probe.py --gpu_idx ${GPU_IDX} --ft_strategy bn+last-3
python linear_probe.py --gpu_idx ${GPU_IDX} --ft_strategy bn_only
python linear_probe.py --gpu_idx ${GPU_IDX} --ft_strategy bn_nolinear
python linear_probe.py --gpu_idx ${GPU_IDX} --ft_strategy full

