import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            torch.nn.GroupNorm(num_groups=32, num_channels=mid_channels, eps=1e-6, affine=True),
            nn.SiLU()
        )

    def forward(self, x):
        return self.double_conv(x)


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

def Normalize(in_channels):
    return nn.BatchNorm2d(in_channels)

class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        block = [
            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            Normalize(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            Normalize(n_channels),
            nn.ReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, num_down, base_ch=64):
        super().__init__()
        assert num_down >= 1, "num_down must be >= 1"

        self.deconv = DoubleConv(in_channels,64)
        
        chs = [base_ch * (2 ** i) for i in range(0,num_down)]
        chs = [chs[i] if i < 2 else chs[i-1] for i in range(len(chs))]
        self.out_channels_per_level = chs 
        

        self.blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        
        for i in range(num_down-1):
            in_ch = chs[i]
            out_ch = chs[i+1]
            self.blocks.append(Down(in_ch, out_ch))
            self.res_blocks.append(ResidualBlock(out_ch))

            
    def forward(self, x):
        feats = []
        x = self.deconv(x)
        feats.append(x)   
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            x = self.res_blocks[i](x)              
            feats.append(x) 
        return feats  


class Decoder(nn.Module):
    def __init__(self, chs, out_channels=1):

        super().__init__()
        assert len(chs) >= 1
        self.num_down = len(chs)
        
        self.up_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        for i in range(self.num_down - 1, 0, -1):
            self.up_blocks.append(Upsample(in_channels=chs[i], out_channels=chs[i - 1]))
            self.res_blocks.append(ResidualBlock(chs[i - 1]))

        self.final_upsample = nn.Conv2d(in_channels=chs[0], out_channels=out_channels, kernel_size=1)
        self.out_activation = nn.Sigmoid()

    def forward(self, feats):

        assert isinstance(feats, (list, tuple)) and len(feats) == self.num_down
        y = feats[-1] 

        for k, i in enumerate(range(self.num_down - 1, 0, -1)):
            y = self.up_blocks[k](y)
            y = self.res_blocks[k](y)
            y = y + feats[i - 1]  

        y = self.final_upsample(y)  
        return self.out_activation(y)


class FMM_model(nn.Module):
    def __init__(self, in_channel, num_down):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channel, num_down=num_down, base_ch=64)
        self.decoder = Decoder(chs=self.encoder.out_channels_per_level, out_channels=1)

    def forward(self, x):
        feats = self.encoder(x)         
        out = self.decoder(feats)       
        return out


