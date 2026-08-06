import os
import logging
import time
import glob
import random
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import sys
import cv2
import torch.nn.functional as F

sys.path.append('..')

from models.diffusion_model import DSFM_Diff

from load_data import build_loader_for_Diff
from functions.resample import LossSecondMomentResampler, LossAwareSampler, UniformSampler
from functions import get_optimizer
from functions.losses import noise_estimation_loss
from functions.masking import generate_condition_idx


# from [-1,1] to [0,1]
def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


# get beta
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.num_modality = len(self.config.data.modalities_name)

        if config.diffusion.sampling_schedule == "uni":
            self.schedule_sampler = UniformSampler(self.num_timesteps)
        else:
            self.schedule_sampler = LossSecondMomentResampler(self.num_timesteps)

        # parameters
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        # variance
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config

        train_loader = build_loader_for_Diff(self.config.data.data_store_path, self.config.data.modalities_name,
                                             self.config.data.modalities_target)

        model = DSFM_Diff(config).to(self.device)

        optimizer = get_optimizer(config, model.parameters())

        epoch, step = 0, 0

        # resume_training
        if self.args.resume_training:
            states = torch.load(os.path.join(self.config.train.ckp_point_path, "diff/ckpt.pth"))
            model.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            epoch = states[2]
            step = states[3]

        model.load_encoders()

        # start training
        while True:
            data_start = time.time()
            data_time = 0
            minibatch = config.train.mini_batch_size
            for i, (m_list, m_target) in enumerate(train_loader):
                # 1 M Z H W  -> Z M H W
                m_list = m_list.permute(2, 1, 0, 3, 4).squeeze(2).to(self.device)

                # (1 Z H W)->(Z 1 H W)
                m_target = m_target.permute(1, 0, 2, 3).to(self.device)

                index = 0
                while index < len(m_target):
                    start = index
                    stop = index + minibatch
                    index += minibatch
                    if stop >= len(m_target):
                        stop = len(m_target)

                    m_target_mini = m_target[start:stop]
                    m_list_mini = m_list[start:stop]

                    n = m_target_mini.size(0)
                    data_time += time.time() - data_start

                    step += 1
                    m_target_mini = m_target_mini.to(self.device).float()
                    m_list_mini = m_list_mini.to(self.device).float()

                    e = torch.randn_like(m_target_mini)
                    b = self.betas

                    condition_idx = generate_condition_idx(self.num_modality)

                    t, weights = self.schedule_sampler.sample(m_target_mini.shape[0], self.device)

                    loss = noise_estimation_loss(model, m_target_mini, t, e, b, m_list_mini, condition_idx)

                    if isinstance(self.schedule_sampler, LossAwareSampler):
                        self.schedule_sampler.update_with_local_losses(
                            t, loss.detach()
                        )

                    loss = loss.mean(dim=0)

                    logging.info(
                        f"step: {step}, loss: {loss.mean(dim=0):.3f},data time: {data_time / (i + 1):.3f}"
                    )

                    optimizer.zero_grad()
                    loss.backward()

                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim_Diff.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()

                    if step % self.config.Diff_train.snapshot_freq == 0 or step == 1:
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]

                        ckp_pth = os.path.join(self.config.train.ckp_point_path, "diff")

                        os.makedirs(ckp_pth, exist_ok=True)
                        torch.save(
                            states,
                            os.path.join(ckp_pth, "ckpt_{}.pth".format(step)),
                        )
                        torch.save(states, os.path.join(ckp_pth, "ckpt.pth"))

                    if step == self.config.Diff_train.stop_step:
                        logging.info(
                            f"finish training, step : {step}"
                        )
                        exit()

                    data_start = time.time()

            epoch += 1

    def sample(self):
        test_loader = build_loader_for_Diff(self.config.data.data_store_path, self.config.data.modalities_name,
                                             self.config.data.modalities_target, False)

        model = DSFM_Diff(self.config).to(self.device)
        ckp_pth = os.path.join(self.config.train.ckp_point_path, "diff")

        logging.info('loading parameters...')
        if getattr(self.config.sampling, "ckpt_id", None) is None:
            states = torch.load(
                os.path.join(ckp_pth, "ckpt.pth"),
                map_location=self.device,
            )

        else:
            states = torch.load(
                os.path.join(ckp_pth, "ckpt_{}.pth".format(self.config.sampling.ckpt_id)),
                map_location=self.device,
            )
        model.load_state_dict(states[0], strict=True)

        logging.info('loading parameters successfully')
        logging.info('loading dataset')
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        self.sample_sequence(model, test_loader)

    def sample_sequence(self, model, test_loader):

        ckp_pth = os.path.join(self.args.log_path, "sampling")
        os.makedirs(ckp_pth, exist_ok=True)

        image_index = 0
        minibatch = self.config.train.mini_batch_size
        logging.info('start generating images')
        for i, (m_list, m_target) in enumerate(test_loader):
            m_list = m_list.permute(2, 1, 0, 3, 4).squeeze(2).to(self.device)
            m_target = m_target.permute(1, 0, 2, 3).to(self.device)
            condition_idx = generate_condition_idx(self.num_modality)

            index = 0
            while index < len(m_target):

                start = index
                stop = index + minibatch
                index += minibatch
                if stop >= len(m_target):
                    stop = len(m_target)

                x0_mini = m_target[start:stop]
                condition_mini = m_list[start:stop]

                x0_mini, condition_mini = x0_mini.to(self.device).float(), condition_mini.to(self.device).float()

                # random noise, star noise
                x = torch.randn(
                    (len(x0_mini)),  # batch_size
                    1,  # data.channels
                    224,  # image_size
                    224,  # image_size
                    device=self.device,
                )

                with torch.no_grad():
                    x = self.sample_image(x, model, condition_mini, condition_idx)
                    x = x.to(self.device)
                    x = (x * 255.0).clamp(0, 255).to(torch.uint8)
                    images = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()

                    for mini_index in range(len(images)):
                        cv2.imwrite(os.path.join(ckp_pth, str(image_index) + '.png'), images[mini_index])
                        image_index += 1

    def sample_image(self, x, model, condition, condition_idx):

        skip = 1
        seq = range(0, self.num_timesteps, skip)

        from functions.denoising import ddpm_steps
        xs, x0_preds = ddpm_steps(x, seq, model, self.betas, condition, condition_idx)

        return x0_preds[-1]





