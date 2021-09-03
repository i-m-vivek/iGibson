#!/bin/bash

gpu="0"
reward_type="dense"
pos="fixed"
lr="1e-4"
death="30.0"
num_steps="1024"
run="0"

# log_dir="hrl_reward_"$reward_type"_pos_"$pos"_sgm_arm_world_irs_"$irs"_sgr_"$sgr"_lr_"$lr"_meta_lr_"$meta_lr"_fr_lr_"$fr_lr"_death_"$death"_init_std_"$init_std_dev_xy"_"$init_std_dev_xy"_"$init_std_dev_z"_failed_pnt_"$failed_pnt"_nsteps_"$num_steps"_ext_col_"$ext_col"_6x6_from_scr_"$name"_run_"$run
# echo $log_dir

python -u train_ppo_tabletop.py \
   --use-gae \
   --sim-gpu-id $gpu \
   --pth-gpu-id $gpu \
   --lr $lr \
   --clip-param 0.1 \
   --value-loss-coef 0.5 \
   --num-train-processes 4 \
   --num-eval-processes 1 \
   --num-steps $num_steps \
   --num-mini-batch 1 \
   --num-updates 50000 \
   --use-linear-lr-decay \
   --use-linear-clip-decay \
   --entropy-coef 0.01 \
   --log-interval 1 \
   --experiment-folder "ckpt/tiago_stadium_point_nav_random" \
   --checkpoint-interval 10 \
   --checkpoint-index -1 \
   --config-file "tiago_stadium_point_nav_random.yaml" \
   --num-eval-episodes 1
   --use_base_only
