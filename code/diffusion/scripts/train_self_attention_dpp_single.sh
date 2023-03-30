#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be omitted.
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --job-name=graph_dif_selfatt
#number of independent tasks we are going to start in this script
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=6
#the amount of memory allocated
#SBATCH --mem=16000M
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=24:00:00
#Skipping many options! see man sbatch
# From here on, we can start our program
echo "Conditional UNET experiment:"

echo "Python version:"
python --version

echo "CUDA visible devices:"
echo "$CUDA_VISIBLE_DEVICES"

echo "CUDA version:"
# nvcc --version
# 
# --tensors_to_diffuse edge_sequence
python main.py --dataset_path ../../graphs_h5/ --run_name conditional_unet_dpp_tests \
	--model self_attention --max_epochs 1000 --check_val_every_n_epoch 5 --batch_size 1 \
	--tensors_to_diffuse edge_sequence --pad_length 23 --diffusion_timesteps 1000 --num_workers 4 \
	--log_every_n_steps 10\
	--disable_carbon_tracker --sample_interval 0 --enable_progress_bar False --single_sample

#  --device "cuda" --accelerator "gpu" --devices -1 --strategy "ddp" 