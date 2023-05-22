python main.py --dataset_path ../../graphs_h5/ --run_name ED_new_v3 \
	--model eq_new --max_epochs 10000 --check_val_every_n_epoch 1 --batch_size 2 \
	--tensors_to_diffuse xyz --pad_length 8 --diffusion_timesteps 10000 --num_workers 8 \
	--log_every_n_steps 128 \
	--disable_carbon_tracker --sample_interval 0 --enable_progress_bar True --single_sample --metrics mse \
	--auto_scale_batch_size False --equiv_n_layers 2 --equiv_hidden_dim 4 --cubic
