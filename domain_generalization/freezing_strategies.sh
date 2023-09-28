#!/bin/bash

GPU_IDX=0

python freezing_strategies.py --gpu_idx ${GPU_IDX} --ft_strategy last-1
python freezing_strategies.py --gpu_idx ${GPU_IDX} --ft_strategy linear_probe
python freezing_strategies.py --gpu_idx ${GPU_IDX} --ft_strategy last-2
python freezing_strategies.py --gpu_idx ${GPU_IDX} --ft_strategy last-3
python freezing_strategies.py --gpu_idx ${GPU_IDX} --ft_strategy bn+last-3
python freezing_strategies.py --gpu_idx ${GPU_IDX} --ft_strategy bn_only
python freezing_strategies.py --gpu_idx ${GPU_IDX} --ft_strategy bn_nolinear
python freezing_strategies.py --gpu_idx ${GPU_IDX} --ft_strategy full

