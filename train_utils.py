import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from tqdm import tqdm
from moving_average import moving_average
from strategies import *

@torch.no_grad()
def p_sample_loop(diffusion, noise, extra_args, device, eta=0, samples_to_capture=-1, need_tqdm=True, clip_value=3):
    mode = diffusion.net_.training
    diffusion.net_.eval()
    img = noise
    imgs = []
    iter_ = reversed(range(diffusion.num_timesteps))
    c_step = diffusion.num_timesteps/samples_to_capture
    next_capture = c_step
    if need_tqdm:
        iter_ = tqdm(iter_)
    for i in iter_:
        img = diffusion.p_sample(
            img,
            torch.full((img.shape[0],), i, dtype=torch.int64).to(device),
            extra_args,
            eta=eta,
            clip_value=clip_value
        )
        if diffusion.num_timesteps - i > next_capture:
            imgs.append(img)
            next_capture += c_step
    imgs.append(img)
    diffusion.net_.train(mode)
    return imgs


def make_none_args(img, label, device):
    return {}


def default_iter_callback(N, loss, last=False):
    None


def make_visualization_(diffusion, device, image_size, need_tqdm=False, eta=0, clip_value=1.2):
    extra_args = {}
    noise = torch.randn(image_size, device=device)
    imgs = p_sample_loop(diffusion, noise, extra_args, "cuda", samples_to_capture=5, need_tqdm=need_tqdm, eta=eta, clip_value=clip_value)
    images_ = []
    for images in imgs:
        images = images.split(1, dim=0)
        images = torch.cat(images, -1)
        images_.append(images)
    images_ = torch.cat(images_, 2)
    return images_


def make_visualization(diffusion, device, image_size, need_tqdm=False, eta=0, clip_value=1.2):
    images_ = make_visualization_(diffusion, device, image_size, need_tqdm=need_tqdm, eta=eta, clip_value=clip_value)
    images_ = images_[0].permute(1, 2, 0).cpu().numpy()
    images_ = (255 * (images_ + 1) / 2).clip(0, 255).astype(np.uint8)
    return images_


def make_iter_callback(diffusion, device, checkpoint_path, image_size, tensorboard, log_interval, ckpt_interval, need_tqdm=False):
    state = {
        "initialized": False,
        "last_log": None,
        "last_ckpt": None
    }

    def iter_callback(N, loss, last=False):
        from datetime import datetime
        t = datetime.now()
        if True:
            tensorboard.add_scalar("loss", loss, N)
        if not state["initialized"]:
            state["initialized"] = True
            state["last_log"] = t
            state["last_ckpt"] = t
            return
        if ((t - state["last_ckpt"]).total_seconds() / 60 > ckpt_interval) or last:
            torch.save({"G": diffusion.net_.state_dict(), "n_timesteps": diffusion.num_timesteps, "time_scale": diffusion.time_scale}, os.path.join(checkpoint_path, f"checkpoint.pt"))
            print("Saved.")
            state["last_ckpt"] = t
        if ((t - state["last_log"]).total_seconds() / 60 > log_interval) or last:
            images_ = make_visualization(diffusion, device, image_size, need_tqdm)
            images_ = cv2.cvtColor(images_, cv2.COLOR_BGR2RGB)
            tensorboard.add_image("visualization", images_, global_step=N, dataformats='HWC')
            tensorboard.flush()
            state["last_log"] = t

    return iter_callback


class InfinityDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, L):
        self.dataset = dataset
        self.L = L

    def __getitem__(self, item):
        idx = random.randint(0, len(self.dataset) - 1)
        r = self.dataset[idx]
        return r

    def __len__(self):
        return self.L


def make_condition(img, label, device):
    return {}


class DiffusionTrain:

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def train(self, train_loader, diffusion, model_ema, model_lr, device, make_extra_args=make_none_args, on_iter=default_iter_callback):
        scheduler = self.scheduler
        total_steps = len(train_loader)
        scheduler.init(diffusion, model_lr, total_steps)
        diffusion.net_.train()
        print(f"Training...")
        pbar = tqdm(train_loader)
        N = 0
        L_tot = 0
        for img, label in pbar:
            scheduler.zero_grad()
            img = img.to(device)
            time = torch.randint(0, diffusion.num_timesteps, (img.shape[0],), device=device)
            extra_args = make_extra_args(img, label, device)
            loss = diffusion.p_loss(img, time, extra_args)
            L_tot += loss.item()
            N += 1
            pbar.set_description(f"Loss: {L_tot / N}")
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion.net_.parameters(), 1)
            scheduler.step()
            moving_average(diffusion.net_, model_ema)
            on_iter(N, loss.item())
            if scheduler.stop(N, total_steps):
                break
        on_iter(N, loss.item(), last=True)


class DiffusionDistillation:

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def train_student_debug(self, distill_train_loader, teacher_diffusion, student_diffusion, student_ema, student_lr, device, make_extra_args=make_none_args, on_iter=default_iter_callback):
        total_steps = len(distill_train_loader)
        scheduler = self.scheduler
        scheduler.init(student_diffusion, student_lr, total_steps)
        teacher_diffusion.net_.eval()
        student_diffusion.net_.train()
        print(f"Distillation...")
        pbar = tqdm(distill_train_loader)
        N = 0
        L_tot = 0

        for img, label in pbar:
            scheduler.zero_grad()
            img = img.to(device)
            time = 2 * torch.randint(0, student_diffusion.num_timesteps, (img.shape[0],), device=device)
            extra_args = make_extra_args(img, label, device)
            loss = teacher_diffusion.distill_loss(student_diffusion, img, time, extra_args)
            L = loss.item()
            L_tot += L
            N += 1
            pbar.set_description(f"Loss: {L_tot / N}")
            loss.backward()
            scheduler.step()
            moving_average(student_diffusion.net_, student_ema)
            if scheduler.stop(N, total_steps):
                break
            on_iter(N, loss.item())
        on_iter(N, loss.item(), last=True)

    def train_student(self, distill_train_loader, teacher_diffusion, student_diffusion, student_ema, student_lr, device, make_extra_args=make_none_args, on_iter=default_iter_callback):
        scheduler = self.scheduler
        total_steps = len(distill_train_loader)
        scheduler.init(student_diffusion, student_lr, total_steps)
        teacher_diffusion.net_.eval()
        student_diffusion.net_.train()
        print(f"Distillation...")
        pbar = tqdm(distill_train_loader)
        N = 0
        L_tot = 0
        for img, label in pbar:
            scheduler.zero_grad()
            img = img.to(device)
            time = 2 * torch.randint(0, student_diffusion.num_timesteps, (img.shape[0],), device=device)
            extra_args = make_extra_args(img, label, device)
            loss = teacher_diffusion.distill_loss(student_diffusion, img, time, extra_args)
            L = loss.item()
            L_tot += L
            N += 1
            pbar.set_description(f"Loss: {L_tot / N}")
            loss.backward()
            scheduler.step()
            moving_average(student_diffusion.net_, student_ema)
            if scheduler.stop(N, total_steps):
                break
            on_iter(N, loss.item())
        on_iter(N, loss.item(), last=True)
