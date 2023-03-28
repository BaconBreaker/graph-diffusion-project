#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be omitted.
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --job-name=graph_dif_selfatt
#number of independent tasks we are going to start in this script
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#the amount of memory allocated
#SBATCH --mem=16000M
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=8:00:00
#Skipping many options! see man sbatch
# From here on, we can start our program
echo "Generating GIFs"

echo "Python version:"
python --version

echo "CUDA visible devices:"
echo "$CUDA_VISIBLE_DEVICES"

echo "CUDA version:"
nvcc --version

# --tensors_to_diffuse edge_sequence
python generate_gif.py --dataset_path ../../graphs_fixed_num_135/ --run_name self_attention_checkpoint \
	--model self_attention --max_epochs 100 --check_val_every_n_epoch 5 --batch_size 2 \
	--tensors_to_diffuse edge_sequence --pad_length 135 --diffusion_timesteps 1000 --num_workers 4 \
  --device "cuda" --accelerator "gpu" --devices -1 --disable_carbon_tracker \
	--t_skips 10 --checkpoint_path "checkpoints/conditional_unet_dpp_tests/epoch=454-val_loss=124584.906.ckpt"
