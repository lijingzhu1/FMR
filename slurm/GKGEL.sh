#!/bin/bash
python main.py -dataset ENTITY -lifelong_name GKGEL -seed 42 -regular_weight 2 -reconstruct_weight 0.01 \
>~/projects/LKGE/LKGE_relation/slurm/results/Entity_GKGEL_rewc_rl_2_rgl_0.01.log 2>~/projects/LKGE/LKGE_relation/slurm/errors/Entity_GKGEL_rewc_rl_2_rgl_0.01_error.log