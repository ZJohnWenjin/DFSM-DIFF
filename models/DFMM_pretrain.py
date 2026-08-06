import torch
from torch import nn
import logging
import torch.nn.functional as F
import os
import numpy as np
import cv2 as cv


from models.load_spec_model import FMM_model, Encoder, Decoder
from load_data import build_loader_for_FMM
from functions.masking import random_zero_along_dim


class FMM_Pretrain():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_modality = len(cfg.data.modalities_name)
        self.model_spe_list = [FMM_model(cfg.model.in_channels, cfg.model.down_num) for i in range(self.num_modality)]
        self.model_map_list = [FMM_model(self.num_modality - 1, cfg.model.down_num) for i in range(self.num_modality)]
        self.model_spe_enc_list = [Encoder(cfg.model.in_channels, cfg.model.down_num) for i in range(self.num_modality)]

    def pretrain(self, specific_encoder=True):
        # pretrain specific and mapping encoder
        for index_model in range(self.num_modality):
            name_of_modality = self.cfg.data.modalities_name[index_model].split('.')[0]

            if specific_encoder:
                model = self.model_spe_list[index_model].to(self.device)
                optim = torch.optim.AdamW(model.parameters(), lr=self.cfg.optim_FMM.lr,
                                          weight_decay=self.cfg.optim_FMM.weight_decay)
                self.train_spe_loop(model, optim, index_model, name_of_modality)
            else:
                map_model = self.model_map_list[index_model].to(self.device)
                spe_enc = self.model_spe_enc_list[index_model].to(self.device)
                optim = torch.optim.AdamW(map_model.encoder.parameters(), lr=self.cfg.optim_FMM.lr,
                                          weight_decay=self.cfg.optim_FMM.weight_decay)
                self.train_mapping_loop(map_model, spe_enc, optim, index_model, name_of_modality)

    def train_spe_loop(self, model, optim, index_model, name_of_modality):
        # pretrain specific encoder
        logging.info('Training specific encoder and decoder for {}'.format(name_of_modality))
        data_loader = self.build_dataloader()

        for epoch in range(1, self.cfg.train.epochs + 1):
            for iter, (inputs_vol_stack) in enumerate(data_loader):
                # 1 M Z H W -> M Z 1 H W
                inputs_vol_stack = inputs_vol_stack.permute(1, 2, 0, 3, 4).to(self.device)
                inputs_vol = inputs_vol_stack[index_model]

                loss_for_vol = 0.0
                index_slice = 0
                acc_chunk = 0
                minibatch = self.cfg.train.mini_batch_size
                while index_slice < len(inputs_vol):
                    acc_chunk += 1
                    start = index_slice
                    stop = index_slice + minibatch
                    index_slice += minibatch
                    if stop >= len(inputs_vol):
                        stop = len(inputs_vol)

                    inputs = inputs_vol[start:stop]
                    output = model(inputs)

                    loss = F.l1_loss(output, inputs, reduction='none').mean()
                    loss_for_vol += loss.item()

                    optim.zero_grad()
                    loss.backward()
                    optim.step()


                logging.info('epoch: {}, vol index: {}, loss: {}'.format(epoch, iter, loss_for_vol / acc_chunk))

            if epoch % self.cfg.train.snapshot_freq_epoch == 0:
                self.save_ckp(model.encoder, model.decoder, name_of_modality, epoch, is_specific_encoder=True)

    def train_mapping_loop(self, map_model, spe_enc, optim, index_model, name_of_modality):
        # pretrain mapping encoder
        logging.info('Training mapping encoder for {}'.format(name_of_modality))
        data_loader = self.build_dataloader()

        map_model, spe_enc = self.load_spe_enc_decoder_parameter(map_model, spe_enc, name_of_modality)
        logging.info('loading spe encoder and decoder for {}'.format(name_of_modality))

        for epoch in range(1, self.cfg.train.epochs + 1):

            for iter, (inputs_vol_stack) in enumerate(data_loader):
                inputs_vol_stack = inputs_vol_stack.permute(1, 2, 0, 3, 4).to(self.device)

                label_vol = inputs_vol_stack[index_model]

                input_modalities = torch.cat((
                    inputs_vol_stack[:index_model],
                    inputs_vol_stack[index_model + 1:]
                ), dim=0).squeeze(2).permute(1, 0, 2, 3)

                input_modalities = random_zero_along_dim(input_modalities)

                loss_for_vol = 0.0
                loss_mse_for_vol = 0.0
                index_slice = 0
                acc_chunk = 0
                minibatch = self.cfg.train.mini_batch_size
                while index_slice < len(label_vol):
                    acc_chunk += 1
                    start = index_slice
                    stop = index_slice + minibatch
                    index_slice += minibatch
                    if stop >= len(label_vol):
                        stop = len(label_vol)

                    inputs = input_modalities[start:stop]
                    label = label_vol[start:stop]

                    spec_en_output = spe_enc(label)
                    map_en_output = map_model.encoder(inputs)
                    output = map_model.decoder(map_en_output)

                    gam = 2.0
                    distill_weights = torch.tensor([gam ** i for i in range(len(spec_en_output))], dtype=torch.float).to(spec_en_output[0].device)
                    distill_weights = distill_weights / distill_weights.sum()
                    loss = sum(w * F.mse_loss(fs, fm) for w, fs, fm in zip(distill_weights, spec_en_output, map_en_output))

                    loss_mse_for_vol += loss.item()
                    loss += F.l1_loss(output, label, reduction='none').mean() * 2
                    loss_for_vol += loss.item()

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                logging.info('epoch: {}, vol index: {}, loss: {}'.format(epoch, iter, loss_for_vol / acc_chunk))

            if epoch % self.cfg.train.snapshot_freq_epoch == 0:
                self.save_ckp(map_model.encoder, map_model.decoder, name_of_modality, epoch, is_specific_encoder=False)

    def load_spe_enc_decoder_parameter(self, map_model, spe_encoder, name_of_modality):
        # loading specific encoder
        encoder_path = os.path.join(self.cfg.train.ckp_point_path, name_of_modality, "specific_encoder",
                                    f"model_epoch{self.cfg.train.decoder_ckp_for_load}.pth")
        ckp_encoder = torch.load(encoder_path, map_location='cpu')
        spe_encoder.load_state_dict(ckp_encoder['model_state_dict'])
        for param in spe_encoder.parameters():
            param.requires_grad = False
        spe_encoder.eval()

        # loading specific decoder
        decoder_path = os.path.join(self.cfg.train.ckp_point_path, name_of_modality, "decoder",
                                    f"model_epoch{self.cfg.train.decoder_ckp_for_load}.pth")
        ckp_decoder = torch.load(decoder_path, map_location='cpu')

        map_model.decoder.load_state_dict(ckp_decoder['model_state_dict'])
        for param in map_model.decoder.parameters():
            param.requires_grad = False

        return map_model, spe_encoder

    def build_dataloader(self):
        return build_loader_for_FMM(self.cfg.data.data_store_path, self.cfg.data.modalities_name)

    def save_ckp(self, encoder, decoder, name_modality, epoch, is_specific_encoder=True):
        if is_specific_encoder:
            # decoder
            save_dir_decoder = os.path.join(self.cfg.train.ckp_point_path, name_modality, "decoder")
            os.makedirs(save_dir_decoder, exist_ok=True)
            save_path_decoder = os.path.join(save_dir_decoder, f"model_epoch{epoch}.pth")

            ckp_decoder = {
                'epoch': epoch,
                'model_state_dict': decoder.state_dict()
            }
            torch.save(ckp_decoder, save_path_decoder)
            logging.info('saving specific encoder and decoder for epoch: {}'.format(epoch))
            encoder_dir = "specific_encoder"
        else:
            encoder_dir = "mapping_encoder"

        # encoder
        save_dir_encoder = os.path.join(self.cfg.train.ckp_point_path, name_modality, encoder_dir)
        os.makedirs(save_dir_encoder, exist_ok=True)
        save_path_encoder = os.path.join(save_dir_encoder, f"model_epoch{epoch}.pth")

        ckp_encoder = {
            'epoch': epoch,
            'model_state_dict': encoder.state_dict()
        }
        torch.save(ckp_encoder, save_path_encoder)
        logging.info('saving {} for epoch: {}'.format(encoder_dir, epoch))

