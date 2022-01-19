from train_utils import *
from unet_ddpm import UNet
from celeba_dataset import CelebaWrapper

BASE_NUM_STEPS = 1024
BASE_TIME_SCALE = 1

def make_model():
    net = UNet(in_channel = 3,
        channel = 128-16,
        channel_multiplier = [1, 2, 2, 4, 4],
        n_res_blocks = 2,
        attn_strides = [8, 16],
        attn_heads = 4,
        use_affine_time = True,
        dropout = 0,
        fold = 1)
    net.image_size = [1, 3, 256, 256]
    return net

def make_dataset():
    return CelebaWrapper(dataset_dir="./data/celeba_256/", resolution=256)

