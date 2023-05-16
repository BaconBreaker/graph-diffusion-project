#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be omitted.
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --job-name=gen_process
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
python ovito_complete_gif.py --dataset_path ../../graphs_h5/ --run_name eq_cond_tanh_pdf_1024 \
	--model equivariant --batch_size 1 \
	--tensors_to_diffuse xyz --pad_length 135 --diffusion_timesteps 1000 \
	--device "cuda" --accelerator "gpu" --devices -1 --disable_carbon_tracker \
	--t_skips 5 --checkpoint_path "checkpoints/ED_cond_tanh_pdf_1024_best_model/epoch=918-val_loss=0.419.ckpt" \
	--equiv_hidden_dim 256 --equiv_n_layers 9 --equiv_pdf_hidden_dim 1024 \
	--fix_noise --single_sample