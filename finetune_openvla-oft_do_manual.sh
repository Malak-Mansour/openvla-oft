#!/bin/bash
#SBATCH --job-name=openvla-oft-finetune
#SBATCH --output=logs/openvla-oft-finetune-%j.out
#SBATCH --error=logs/openvla-oft-finetune-%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos        
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-06             # 🚨 Use idle node
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00


source ~/.bashrc
cd /l/users/malak.mansour/ICL/openvla-oft
conda activate openvla-oft


torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /l/users/malak.mansour/Datasets/do_manual/rlds \
  --dataset_name do_manual \
  --run_root_dir /l/users/malak.mansour/OpenVLA/runs/do_manual \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "mim7995-mbzuai" \
  --wandb_project "openvla-oft-do_manual" \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state


# launch with 
#         bash finetune_openvla-oft_do_manual.sh
#         sbatch finetune_openvla-oft_do_manual.sh
