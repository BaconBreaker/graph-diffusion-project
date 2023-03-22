#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be omitted.
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --job-name=graph_dif_cond_unet
#number of independent tasks we are going to start in this script
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#the amount of memory allocated
#SBATCH --mem=12000M
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=8:00:00
#Skipping many options! see man sbatch
# From here on, we can start our program
echo "Python version:"
python --version

echo "CUDA visible devices:"
echo "$CUDA_VISIBLE_DEVICES"

echo "CUDA version:"
nvcc --version

echo "Conditional UNET experiment:"

python main.py --dataset_path ../../graphs_fixed_num_135/ --run_name 123 --max_epochs 10 --check_val_every_n_epoch 1 --batch_size 16 --tensors_to_diffuse edge_sequence --pad_length 135 --diffusion_timesteps 100 --num_workers 4 --log_every_n_steps 10 --disable_carbon_tracker