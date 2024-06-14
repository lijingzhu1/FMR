#!/bin/bash
python main.py -dataset FACT -lifelong_name GKGEL -seed 42 -regular_weight 1 -reconstruct_weight 0.1 \
>~/projects/LKGE/LKGE_relation/slurm/results/Fact_GKGEL_rewc_rl_0.1_rgl_1.log 2>~/projects/LKGE/LKGE_relation/slurm/errors/Fact_GKGEL_rewc_rl_0.1_rgl_1_error.log