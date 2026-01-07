
import torch
import torch.nn as nn
import math
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.nn import functional as F


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'none':
        return lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=1.0)
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count_start - opt.n_epochs) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cos_wr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

def define_G(opt, input_nc, output_nc, which_model_netG, encoder_norm='instance', decoder_norm='instance', bottleneck_norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[], input_scan_types=['T1w','T2w','FLAIR'], global_rescaling_T1w_bounds=(0.0, 1.0), global_rescaling_T2w_bounds=(0.0, 1.0), global_rescaling_FLAIR_bounds=(0.0, 1.0), output_activation='sigmoid'):
    if which_model_netG == 'res_unet':
        netG = ResUnet(input_nc, output_nc, output_activation=output_activation, norm_type=opt.norm_G)
    elif which_model_netG == 'unet':
        netG = UNet(input_nc, output_nc, norm_type=opt.norm_G)
    elif which_model_netG == 'shared_attn_unet_drop':
        netG = SharedAttnUNetDrop(output_channels=output_nc, feature_sizes=opt.feature_channels_G, norm_type=opt.norm_G, dropout_prob_bottleneck=opt.dropout_prob_bottleneck, dropout_prob_decoder=opt.dropout_prob_decoder)
    elif which_model_netG == 'shared_attn_unet':
        netG = SharedAttnUNet(output_channels=output_nc, feature_sizes=opt.feature_channels_G, encoder_norm=encoder_norm, decoder_norm=decoder_norm, bottleneck_norm=bottleneck_norm)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


## ResUnet ##
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding, norm_layer):
        super(ResidualConv, self).__init__()
        self.conv_block = nn.Sequential(
            norm_layer(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            norm_layer(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            norm_layer(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_out,
        filters=[64, 128, 256, 512],
        output_activation='sigmoid',
        norm_type='instance'   # <-- 'batch' or 'instance'
    ):
        super(ResUnet, self).__init__()

        # Choose normalization layer factory based on norm_type
        if norm_type == 'batch':
            norm_layer = lambda num_features: nn.BatchNorm2d(num_features)
        elif norm_type == 'instance':
            norm_layer = lambda num_features: nn.InstanceNorm2d(num_features, affine=True)
        else:
            raise ValueError("norm_type must be 'batch' or 'instance'")

        # Input block: first conv -> norm -> ReLU -> conv
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel_in, filters[0], kernel_size=3, padding=1),
            norm_layer(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        # Skip connection for the input block
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel_in, filters[0], kernel_size=3, padding=1)
        )

        # Encoder: ResidualConv layers
        self.residual_conv_1 = ResidualConv(filters[0], filters[1], stride=2, padding=1, norm_layer=norm_layer)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], stride=2, padding=1, norm_layer=norm_layer)

        # Bridge
        self.bridge = ResidualConv(filters[2], filters[3], stride=2, padding=1, norm_layer=norm_layer)

        # Decoder: upsample + residual conv
        self.upsample_1 = Upsample(filters[3], filters[3], kernel=2, stride=2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], stride=1, padding=1, norm_layer=norm_layer)

        self.upsample_2 = Upsample(filters[2], filters[2], kernel=2, stride=2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], stride=1, padding=1, norm_layer=norm_layer)

        self.upsample_3 = Upsample(filters[1], filters[1], kernel=2, stride=2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], stride=1, padding=1, norm_layer=norm_layer)

        # Output block: 1×1 conv + activation
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], channel_out, kernel_size=1, stride=1),
            nn.ReLU() if output_activation == 'relu'
            else nn.Sigmoid() if output_activation == 'sigmoid'
            else nn.Tanh() if output_activation == 'tanh'
            else nn.Identity()
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        # Bridge
        x4 = self.bridge(x3)

        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)
        return output

## UNet ##
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual_connection(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)
    

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(EncoderBlock, self).__init__()
        self.residual_block = ResidualBlock(in_channels, out_channels, norm_layer)
        self.strided_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.residual_block(x)
        pooled = self.strided_conv(x)
        return x, pooled


class DecoderBlockDrop(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, dropout_prob: float = 0.0):
        super(DecoderBlockDrop, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),   
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat((x, skip_connection), dim=1)
        return self.conv(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DecoderBlock, self).__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        conv_in_channels = out_channels * 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        # x: [Batch, in_channels, H, W]
        x = self.upconv(x)
        # x: [Batch, out_channels, 2H, 2W]

        x = torch.cat((x, skip_connection), dim=1)
        
        return self.conv(x)


def make_norm(kind: str):
    """
    kind ∈ {'batch', 'instance', 'group', 'none'}
    Returns a callable  f(channels) → nn.Module
    """
    if kind == 'batch':
        return lambda c: nn.BatchNorm2d(c, affine=True)
    if kind == 'instance':
        return lambda c: nn.InstanceNorm2d(c, affine=True)
    if kind == 'group':                   # GroupNorm with one group == LayerNorm for CNNs
        return lambda c: nn.GroupNorm(1, c, affine=True)
    if kind == 'none':
        return lambda c: nn.Identity()
    raise ValueError(f'Unknown norm kind: {kind}')

class UNet(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        feature_sizes=[64, 128, 256],
        norm_type='instance'   # 'batch' or 'instance'
    ):
        super(UNet, self).__init__()

        norm = make_norm(norm_type)   

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Encoder
        for i, feature_size in enumerate(feature_sizes):
            in_ch = input_channels if i == 0 else feature_sizes[i-1]
            self.encoder_blocks.append(
                EncoderBlock(in_ch, feature_size, norm)
            )

        # Bottleneck
        bottleneck_channels = feature_sizes[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_sizes[-1], bottleneck_channels, kernel_size=3, padding=1),
            norm(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            norm(bottleneck_channels),
            nn.ReLU(inplace=True)
        )

        # Decoder
        for i in range(len(feature_sizes)-1, -1, -1):
            if i == len(feature_sizes)-1:
                in_ch = bottleneck_channels
            else:
                in_ch = feature_sizes[i+1]
            out_ch = feature_sizes[i]
            self.decoder_blocks.append(
                DecoderBlock(in_ch, out_ch, norm)
            )

        # Final Convolution
        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_sizes[0], output_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for encoder in self.encoder_blocks:
            x, pooled = encoder(x)
            skip_connections.append(x)
            x = pooled

        skip_connections = skip_connections[::-1]

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, decoder in enumerate(self.decoder_blocks):
            x = decoder(x, skip_connections[i])

        # Final output
        return self.output_layer(x)
    

## Unet shared encoder with attention fusion ##
    
class AttentionFusion(nn.Module):
    """Permutation-invariant, modality-aware fusion with shared MLP attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        
        # Shared MLP applied to all modalities
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels)
        )

    def forward(self, feats):
        """
        feats: list of feature maps, each B×C×H×W, variable length
        """
        B, C, H, W = feats[0].shape
        M = len(feats)
        

        # Global average pooling per modality
        summaries = torch.stack([f.mean(dim=(2, 3)) for f in feats], dim=1)  # B × M × C

        # Shared MLP to compute logits per modality
        logits = self.mlp(summaries)  # B × M × C

        # Softmax over modalities
        w = torch.softmax(logits, dim=1)  # B × M × C

        # Weighted sum of feature maps
        out = torch.zeros_like(feats[0])
        for m, f in enumerate(feats):
            weight = w[:, m].unsqueeze(-1).unsqueeze(-1)  # B × C × 1 × 1
            out = out + weight * f

        return out
    

class SharedAttnUNet(nn.Module):
    """U‑Net with shared encoder weights and attention fusion, matching depth of standard U‑Net."""
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 3,
        feature_sizes: tuple[int, ...] = (32, 64, 128),
        encoder_norm='instance', decoder_norm='batch', bottleneck_norm='batch',
    ):
        super().__init__()
        self._enc_norm = make_norm(encoder_norm)
        self._bnk_norm = make_norm(bottleneck_norm)
        self._dec_norm = make_norm(decoder_norm)

        # Shared encoder (one per feature size)
        self.enc_blocks = nn.ModuleList()
        in_ch = input_channels
        for fs in feature_sizes:
            self.enc_blocks.append(EncoderBlock(in_ch, fs, self._enc_norm))
            in_ch = fs

        # Bottleneck doubles channels like standard UNet
        bottleneck_ch = feature_sizes[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_sizes[-1], bottleneck_ch, 3, padding=1),
            self._bnk_norm(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1),
            self._bnk_norm(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        # Attention fusers for skip connections and bottleneck
        self.skip_fusers = nn.ModuleList([AttentionFusion(fs) for fs in feature_sizes])
        self.bottleneck_fuser = AttentionFusion(bottleneck_ch)

        # Decoder blocks: reverse feature sizes, including the doubled bottleneck
        self.dec_blocks = nn.ModuleList()
        prev_ch = bottleneck_ch
        for fs in reversed(feature_sizes):
            self.dec_blocks.append(DecoderBlock(prev_ch, fs, self._dec_norm))
            prev_ch = fs

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_sizes[0], output_channels, 1),
            nn.Sigmoid(),
        )

    def _encode_one(self, x):
        skips = []
        for enc in self.enc_blocks:
            s, x = enc(x)
            skips.append(s)
        x = self.bottleneck(x)
        return skips, x

    def forward(self, inputs):
        # Accept list or dict of single‑channel tensors
        if isinstance(inputs, dict):
            inputs = list(inputs.values())

        # Collect per‑modality skips and bottlenecks
        all_skips = [[] for _ in self.enc_blocks]
        all_bottlenecks = []
        for x in inputs:
            skips, bott = self._encode_one(x)
            for i, s in enumerate(skips):
                all_skips[i].append(s)
            all_bottlenecks.append(bott)

        # Fuse skips and bottleneck
        fused_skips = [f(user_skips) for f, user_skips in zip(self.skip_fusers, all_skips)]
        fused_bottleneck = self.bottleneck_fuser(all_bottlenecks)

        # Decode
        x = fused_bottleneck
        for dec, skip in zip(self.dec_blocks, reversed(fused_skips)):
            x = dec(x, skip)

        return self.output_layer(x)


class SharedAttnUNetDrop(nn.Module):
    """U-Net with shared encoder weights and attention fusion"""
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 3,
        feature_sizes: tuple[int, ...] = (32, 64, 128),
        norm_type: str = "instance",
        dropout_prob_bottleneck: float = 0.0,
        dropout_prob_decoder: float = 0.0,
    ):
        super().__init__()

        # Basic validation
        for name_, p_ in (("bottleneck", dropout_prob_bottleneck),
                          ("decoder", dropout_prob_decoder)):
            if not (0.0 <= p_ <= 1.0):
                raise ValueError(f"Dropout prob for {name_} must be in [0, 1], got {p_}.")

        norm = make_norm(norm_type)

        # Shared encoder (one per feature size)
        self.enc_blocks = nn.ModuleList()
        in_ch = input_channels
        for fs in feature_sizes:
            self.enc_blocks.append(EncoderBlock(in_ch, fs, norm))
            in_ch = fs

        # Bottleneck doubles channels like standard UNet
        bottleneck_ch = feature_sizes[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_sizes[-1], bottleneck_ch, 3, padding=1),
            norm(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob_bottleneck),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1),
            norm(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob_bottleneck),
        )

        # Attention fusers for skip connections and bottleneck
        self.skip_fusers = nn.ModuleList([AttentionFusion(fs) for fs in feature_sizes])
        self.bottleneck_fuser = AttentionFusion(bottleneck_ch)

        # Decoder blocks: reverse feature sizes, including the doubled bottleneck
        self.dec_blocks = nn.ModuleList()
        prev_ch = bottleneck_ch
        for fs in reversed(feature_sizes):
            self.dec_blocks.append(DecoderBlockDrop(prev_ch, fs, norm, dropout_prob_decoder))
            prev_ch = fs

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_sizes[0], output_channels, 1),
            nn.Sigmoid(),
        )

    def _encode_one(self, x):
        skips = []
        for enc in self.enc_blocks:
            s, x = enc(x)
            skips.append(s)
        x = self.bottleneck(x)
        return skips, x

    def forward(self, inputs):
        # Accept list or dict of single-channel tensors
        if isinstance(inputs, dict):
            inputs = list(inputs.values())

        # Collect per-modality skips and bottlenecks
        all_skips = [[] for _ in self.enc_blocks]
        all_bottlenecks = []
        for x in inputs:
            skips, bott = self._encode_one(x)
            for i, s in enumerate(skips):
                all_skips[i].append(s)
            all_bottlenecks.append(bott)

        # Fuse skips and bottleneck
        fused_skips = [f(user_skips) for f, user_skips in zip(self.skip_fusers, all_skips)]
        fused_bottleneck = self.bottleneck_fuser(all_bottlenecks)

        # Decode
        x = fused_bottleneck
        for dec, skip in zip(self.dec_blocks, reversed(fused_skips)):
            x = dec(x, skip)

        return self.output_layer(x)
    
    
    