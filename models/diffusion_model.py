import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.load_spec_model import Encoder, Upsample


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x):
    return x * torch.sigmoid(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Normalize(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Normalize(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Diff_ResnetBlock(nn.Module):
    def __init__(
            self,
            channels,
            out_channels,
            emb_channels=64,
            dropout=False,
            use_conv=True,
            use_scale_shift_norm=True,
            use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            Normalize(channels),
            nn.SiLU(),
            torch.nn.Conv2d(channels,
                            self.out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            Normalize(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            torch.nn.Conv2d(self.out_channels,
                            self.out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = torch.nn.Conv2d(channels,
                                                   self.out_channels,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
        else:
            self.skip_connection = torch.nn.Conv2d(channels,
                                                   self.out_channels,
                                                   kernel_size=1,
                                                   stride=1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h



class AttnBlock(nn.Module):
    def __init__(self, in_channels, head=32):
        super().__init__()
        self.in_channels = in_channels
        self.head = head

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, kv):
        h_ = x
        h_, kv = self.norm(h_), self.norm(kv)
        q = self.q(h_)
        k = self.k(kv)
        v = self.v(kv)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b * self.head, c // self.head, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b * self.head, c // self.head, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b * self.head, c // self.head, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_



class Diff_Encoder(nn.Module):
    def __init__(self, in_channels, num_down, emb_channels=64, base_ch=64,
                 dropout=0.0,
                 use_conv=True,
                 use_scale_shift_norm=True,
                 use_checkpoint=False):
        super().__init__()
        assert num_down >= 1, "num_down must be >= 1"

        self.num_down = num_down
        self.deconv = DoubleConv(in_channels, 64)


        chs = [base_ch * (2 ** i) for i in range(0, num_down)]
        chs = [chs[i] if i < 2 else chs[i - 1] for i in range(len(chs))]
        self.out_channels_per_level = chs

        self.down_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        self.attnBlock = AttnBlock(chs[-2])

        for i in range(num_down - 1):
            in_ch = chs[i]
            out_ch = chs[i + 1]

            self.down_blocks.append(Down(in_ch, out_ch))

            self.res_blocks.append(
                Diff_ResnetBlock(
                    channels=out_ch,
                    out_channels=out_ch,
                    emb_channels=emb_channels,
                    dropout=dropout,
                    use_conv=use_conv,
                    use_scale_shift_norm=use_scale_shift_norm,
                    use_checkpoint=use_checkpoint,
                )
            )

    def forward(
            self,
            x,
            emb,
            condition_features=None,
            csmm_blocks=None,
    ):
        """
        condition_features:
            [
                [modality_1_level_0, modality_2_level_0, ...],
                [modality_1_level_1, modality_2_level_1, ...],
            ]
        """

        feats = []

        # Level 0 backbone feature
        x = self.deconv(x)

        if condition_features is not None:
            if csmm_blocks is None:
                raise ValueError(
                    "csmm_blocks must be provided when "
                    "condition_features are provided."
                )

            x = csmm_blocks[0](
                backbone_feature=x,
                modality_features=condition_features[0],
            )

        feats.append(x)

        # Level 1, 2, ..., L
        for i in range(len(self.down_blocks)):

            x = self.down_blocks[i](x)
            x = self.res_blocks[i](x, emb)

            level_idx = i + 1

            if condition_features is not None:
                x = csmm_blocks[level_idx](
                    backbone_feature=x,
                    modality_features=condition_features[level_idx],
                )

            feats.append(x)

        return feats


class Diff_Decoder(nn.Module):
    def __init__(
            self,
            chs,
            out_channels=1,
            emb_channels=64,
            dropout=0.0,
            use_conv=True,
            use_scale_shift_norm=True,
            use_checkpoint=False,
    ):
        super().__init__()
        self.num_levels = len(chs)
        self.up_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        self.level_to_res_idx = {}



        for i in range(self.num_levels - 1, 0, -1):
            dec_ch = chs[i] 
            skip_ch = chs[i - 1]  

            self.up_blocks.append(
                Upsample(in_channels=dec_ch, out_channels=skip_ch)
            )

            self.res_blocks.append(
                Diff_ResnetBlock(
                    channels=skip_ch,
                    out_channels=skip_ch,
                    emb_channels=emb_channels,
                    dropout=dropout,
                    use_conv=use_conv,
                    use_scale_shift_norm=use_scale_shift_norm,
                    use_checkpoint=use_checkpoint,
                )
            )


        self.final_upsample = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    def forward(
            self,
            feats,
            emb,
            condition_features=None,
            csmm_blocks=None,
    ):
        """
        feats:
            Diffusion encoder features

        condition_features:
            Layer-wise modality features returned by get_condition()
        """

        y = feats[-1]

        for k, i in enumerate(
                range(self.num_levels - 1, 0, -1)
        ):
            skip = feats[i - 1]

            # 上采样并融合 skip connection
            y = self.up_blocks[k](y) + skip

            # 当前 decoder stage 的特征
            y = self.res_blocks[k](y, emb)

            # 当前 decoder 输出对应的实际 level
            #
            # 例如三个层：
            # i=2 -> target_level=1
            # i=1 -> target_level=0
            target_level = i - 1

            if condition_features is not None:
                if csmm_blocks is None:
                    raise ValueError(
                        "csmm_blocks must be provided when "
                        "condition_features are provided."
                    )

                y = csmm_blocks[target_level](
                    backbone_feature=y,
                    modality_features=condition_features[target_level],
                )

        y = self.final_upsample(y)

        return y


class CSMMBlock(nn.Module):
    """
    backbone_feature:
        [B, C_backbone, H, W]

    modality_features:
        [
            [B, C_condition, Hc, Wc],
            [B, C_condition, Hc, Wc],
            ...
        ]
    output:
        [B, C_backbone, H, W]
    """

    def __init__(
        self,
        condition_channels,
        backbone_channels,
        attention_channels=None,
    ):
        super().__init__()

        self.condition_channels = condition_channels
        self.backbone_channels = backbone_channels

        if attention_channels is None:
            attention_channels = min(
                condition_channels,
                backbone_channels
            )

        self.attention_channels = attention_channels

        # Q = Wq(B)
        self.q_proj = nn.Conv2d(
            backbone_channels,
            attention_channels,
            kernel_size=1,
            bias=False,
        )

        # Ki = Wk(Si)
        # 所有模态共享同一个 Wk，符合论文中的 W_k 表达
        self.k_proj = nn.Conv2d(
            condition_channels,
            attention_channels,
            kernel_size=1,
            bias=False,
        )

        # Vi = Wv(Si)
        # 输出通道变成 backbone_channels，方便做残差相加
        self.v_proj = nn.Conv2d(
            condition_channels,
            backbone_channels,
            kernel_size=1,
            bias=False,
        )

        self.scale = attention_channels ** -0.5

    @staticmethod
    def local_feature_selection(feature):
        """
        S_n^l = F_n^l * sigmoid(GAP(F_n^l))

        feature:
            [B, C, H, W]

        channel_weight:
            [B, C, 1, 1]
        """
        channel_weight = F.adaptive_avg_pool2d(
            feature,
            output_size=1,
        )

        channel_weight = torch.sigmoid(channel_weight)

        selected_feature = feature * channel_weight

        return selected_feature

    def forward(
        self,
        backbone_feature,
        modality_features,
        return_weights=False,
    ):
        """
        backbone_feature:
            [B, Cb, H, W]

        modality_features:
            List[[B, Cc, Hc, Wc]]

        modality_weights:
            [B, num_modalities, H, W]
        """

        if not isinstance(modality_features, (list, tuple)):
            raise TypeError(
                "modality_features must be a list or tuple."
            )

        if len(modality_features) == 0:
            raise ValueError(
                "modality_features cannot be empty."
            )


        # Q: [B, d, H, W]
        query = self.q_proj(backbone_feature)

        score_list = []
        value_list = []

        for modality_feature in modality_features:

            if modality_feature.ndim != 4:
                raise ValueError(
                    "Each modality feature must have shape "
                    "[B, C, H, W], but got "
                    f"{tuple(modality_feature.shape)}"
                )

            if modality_feature.shape[0] != backbone_feature.shape[0]:
                raise ValueError(
                    "Batch size of backbone and condition "
                    "features must be identical."
                )


            # Step 1: Local Feature Selection and Re-weighting
            selected_feature = self.local_feature_selection(
                modality_feature
            )

            # Step 2: Backbone-guided Inter-modality Gating
            key = self.k_proj(selected_feature)
            value = self.v_proj(selected_feature)

            # s_i(x,y) = <Q(x,y), K_i(x,y)> / sqrt(d)
            # query/key: [B, d, H, W]
            # score:     [B, 1, H, W]
            score = torch.sum(
                query * key,
                dim=1,
                keepdim=True,
            )

            score = score * self.scale

            score_list.append(score)
            value_list.append(value)

        # [B, num_modalities, H, W]
        modality_scores = torch.cat(
            score_list,
            dim=1,
        )

        modality_weights = torch.softmax(
            modality_scores,
            dim=1,
        )

        # [B, num_modalities, Cb, H, W]
        value_stack = torch.stack(
            value_list,
            dim=1,
        )

        # weights:
        # [B, num_modalities, 1, H, W]
        expanded_weights = modality_weights.unsqueeze(dim=2)

        # F_cond = sum_i w_i * V_i
        # [B, Cb, H, W]
        conditioned_feature = torch.sum(
            expanded_weights * value_stack,
            dim=1,
        )

        # B_out = B_in + F_cond
        output = backbone_feature + conditioned_feature

        if return_weights:
            return output, modality_weights

        return output

class DSFM_Diff(nn.Module):
    def __init__(self, cfg,n_channels=1, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.ch = cfg.model.t_emb

        self.cfg = cfg

        self.num_modality = len(cfg.data.modalities_name)
        self.model_spe_list = nn.ModuleList(
            [Encoder(cfg.model.in_channels, cfg.model.down_num) for _ in range(self.num_modality)]
        )
        self.model_map_list = nn.ModuleList(
            [Encoder(self.num_modality - 1, cfg.model.down_num) for _ in range(self.num_modality)]
        )

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.ch),
            torch.nn.Linear(self.ch,
                            self.ch),
        ])

        self.diff_encoder = Diff_Encoder(cfg.model.in_channels, cfg.model.down_num, emb_channels=cfg.model.t_emb,
                                         base_ch=cfg.model.base_ch)

        feats_ch = self.diff_encoder.out_channels_per_level

        condition_chs = (
            self.model_spe_list[0].out_channels_per_level
        )

        if len(condition_chs) != len(feats_ch):
            raise ValueError(
                "Condition encoder and diffusion encoder must have "
                "the same number of feature levels. "
                f"Got {len(condition_chs)} and {len(feats_ch)}."
            )

        self.condition_channels_per_level = condition_chs
        self.backbone_channels_per_level = feats_ch

        # CSMM for encoder
        self.encoder_csmm_list = nn.ModuleList([
            CSMMBlock(
                condition_channels=condition_chs[level],
                backbone_channels=feats_ch[level],
                attention_channels=min(
                    condition_chs[level],
                    feats_ch[level],
                ),
            )
            for level in range(len(feats_ch))
        ])

        # CSMM for decoder
        self.decoder_csmm_list = nn.ModuleList([
            CSMMBlock(
                condition_channels=condition_chs[level],
                backbone_channels=feats_ch[level],
                attention_channels=min(
                    condition_chs[level],
                    feats_ch[level],
                ),
            )
            for level in range(len(feats_ch) - 1)
        ])

        self.diff_decoder = Diff_Decoder(chs=feats_ch, out_channels=1, emb_channels=cfg.model.t_emb)

    def forward(
            self,
            x,
            condition,
            condition_idx,
            t,
    ):

        temb = get_timestep_embedding(
            t,
            self.cfg.model.t_emb,
        )

        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # DFMM condition features
        feature_to_CSMM = self.get_condition(
            condition,
            condition_idx,
        )

        # Diffusion encoder + encoder CSMM
        feats = self.diff_encoder(
            x=x,
            emb=temb,
            condition_features=feature_to_CSMM,
            csmm_blocks=self.encoder_csmm_list,
        )

        # Diffusion decoder + decoder CSMM
        out = self.diff_decoder(
            feats=feats,
            emb=temb,
            condition_features=feature_to_CSMM,
            csmm_blocks=self.decoder_csmm_list,
        )

        return out

    def get_condition(
            self,
            condition,
            condition_idx,
    ):
        if torch.is_tensor(condition_idx):
            condition_idx = condition_idx.detach().cpu().tolist()

        if len(condition_idx) != self.num_modality:
            raise ValueError(
                "condition_idx must have length "
                f"{self.num_modality}, but got "
                f"{len(condition_idx)}."
            )

        inter_feature_list = []

        for i in range(self.num_modality):

            if bool(condition_idx[i]):
                # if modality available
                encoder_input = condition[:, i:i + 1]

                encoder_output = self.model_spe_list[i](
                    encoder_input
                )

            else:
                # use available modalities to map
                encoder_input = torch.cat(
                    [
                        condition[:, :i],
                        condition[:, i + 1:],
                    ],
                    dim=1,
                )

                encoder_output = self.model_map_list[i](
                    encoder_input
                )

            inter_feature_list.append(encoder_output)

        feature_to_CSMM = [
            list(level_features)
            for level_features in zip(*inter_feature_list)
        ]

        return feature_to_CSMM


    def load_encoders(self, ):
        for index_model in range(self.num_modality):
            name_of_modality = self.cfg.data.modalities_name[index_model].split('.')[0]
            spe_encoder = self.model_spe_list[index_model]
            map_encoder = self.model_map_list[index_model]
            self.load_parameter(spe_encoder, name_of_modality, spec=True)
            self.load_parameter(map_encoder, name_of_modality, spec=False)

    def load_parameter(self, encoder, name_of_modality, spec=True):
        if spec:
            dir = "specific_encoder"
        else:
            dir = "mapping_encoder"
        decoder_path = os.path.join(self.cfg.train.ckp_point_path, name_of_modality, dir,
                                    f"model_epoch{self.cfg.train.decoder_ckp_for_load}.pth")
        ckp_decoder = torch.load(decoder_path, map_location='cpu')

        encoder.load_state_dict(ckp_decoder['model_state_dict'])
        logging.info('successfully loading {} for {}'.format(dir, name_of_modality))

        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

