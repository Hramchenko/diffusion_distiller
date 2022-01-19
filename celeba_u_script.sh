# # Train the base model with 1024 diffusion steps

python ./train.py --module celeba_u --name celeba --dname original --batch_size 3 --num_workers 4 --num_iters 150000


# # Model distillation

# ## Distillate to 512 steps
python ./distillate.py --module celeba_u --diffusion GaussianDiffusionDefault --name celeba --dname base_0 --base_checkpoint ./checkpoint_base.pt --batch_size 3 --num_workers 4 --num_iters 5000 --log_interval 5

# ## Distillate to 256 steps
python ./distillate.py --module celeba_u --diffusion GaussianDiffusionDefault --name celeba --dname base_1 --base_checkpoint ./checkpoints/celeba/base_0/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 5000 --log_interval 5

# ## Distillate to 128 steps
python ./distillate.py --module celeba_u --diffusion GaussianDiffusionDefault --name celeba --dname base_2 --base_checkpoint ./checkpoints/celeba/base_1/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 5000 --log_interval 5

# ## Distillate to 64 steps
python ./distillate.py --module celeba_u --diffusion GaussianDiffusionDefault --name celeba --dname base_3 --base_checkpoint ./checkpoints/celeba/base_2/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 5000 --log_interval 5

# ## Distillate to 32 steps
python ./distillate.py --module celeba_u --diffusion GaussianDiffusionDefault --name celeba --dname base_4 --base_checkpoint ./checkpoints/celeba/base_3/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 10000 --log_interval 5

# ## Distillate to 16 steps
python ./distillate.py --module celeba_u --diffusion GaussianDiffusionDefault --name celeba --dname base_5 --base_checkpoint ./checkpoints/celeba/base_4/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 10000 --log_interval 5

# ## Distillate to 8 steps
python ./distillate.py --module celeba_u --diffusion GaussianDiffusionDefault --name celeba --dname base_6 --base_checkpoint ./checkpoints/celeba/base_5/checkpoint.pt --batch_size 3 --num_workers 4 --num_iters 10000 --log_interval 5

# # Image generation
python ./sample.py --out_file ./images/celeba_u_5.png --module celeba_u --checkpoint ./checkpoints/celeba/base_5/checkpoint.pt --batch_size 4 --clip_value 1.0

