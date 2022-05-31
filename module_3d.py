from train_utils import *
from unet_ddpm import UNet
from dataset_3d import NiftiPairImageGenerator
from torchvision.transforms import Compose, Lambda

BASE_NUM_STEPS = 1024
BASE_TIME_SCALE = 1
inputfolder = "../Datasets/Nii-Dataset/input/"
targetfolder = "../Datasets/Nii-Dataset/target/"

transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.unsqueeze(0)),
    Lambda(lambda t: t.transpose(3, 1)),
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.permute(3, 0, 1, 2)),
    Lambda(lambda t: t.transpose(3, 1)),
])

def make_model():
    net = UNet(in_channel = 1+2,#3,
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
    return NiftiPairImageGenerator(
        inputfolder,
        targetfolder,
        input_size=128,
        depth_size=96,
        transform=input_transform, #if with_condition else transform,
        target_transform=transform,
        full_channel_mask=True
    )