import argparse
import importlib
from v_diffusion import make_beta_schedule
from moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
import os
from train_utils import make_visualization
import cv2

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--out_file", help="Path to image.", type=str, default="")
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--time_scale", help="Diffusion time scale.", type=int, default=1)
    parser.add_argument("--eta", help="Amount of random noise on interation(recommended non-zero values only for not distilled model).", type=int, default=0)
    parser.add_argument("--clip_value", help="Noise clipping value.", type=float, default=1.2)
    return parser

def sample_images(args, make_model):
    device = torch.device("cuda")

    teacher_ema = make_model().to(device)

    def make_diffusion(model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("v_diffusion")
        D = getattr(M, args.diffusion)
        return D(model, betas, time_scale=time_scale)

    teacher = make_model().to(device)

    ckpt = torch.load(args.checkpoint)
    teacher.load_state_dict(ckpt["G"])
    n_timesteps = ckpt["n_timesteps"]//args.time_scale
    time_scale = ckpt["time_scale"]*args.time_scale
    del ckpt
    print("Model loaded.")

    teacher_diffusion = make_diffusion(teacher, n_timesteps, time_scale, device)
    image_size = deepcopy(teacher.image_size)
    image_size[0] = args.batch_size

    img = make_visualization(teacher_diffusion, device, image_size, need_tqdm=True, eta=args.eta, clip_value=args.clip_value)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    cv2.imwrite(args.out_file, img)

    print("Finished.")

parser = make_argument_parser()

args = parser.parse_args()

M = importlib.import_module(args.module)
make_model = getattr(M, "make_model")

sample_images(args, make_model)