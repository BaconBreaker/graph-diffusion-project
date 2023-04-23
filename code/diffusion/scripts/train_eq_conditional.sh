#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be omitted.
#SBATCH -p gpu --gres=gpu:titanrtx:2
#SBATCH --job-name=ED_cond
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
#  --strategy "ddp"
python main.py --dataset_path ../../graphs_fixed_num_135/ --run_name ED_cond \
	--model equivariant --max_epochs 10000 --check_val_every_n_epoch 1 --batch_size 8 \
	--tensors_to_diffuse xyz --pad_length 135 --diffusion_timesteps 1000 --num_workers 4 \
	--log_every_n_steps 1 --device cuda --accelerator gpu --devices -1 --strategy ddp \
	--disable_carbon_tracker --sample_interval 0 --enable_progress_bar True \
	--equiv_hidden_dim 256 --equiv_n_layers 9 --conditional --fast_dev_run
