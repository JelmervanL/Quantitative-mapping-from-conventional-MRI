import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np
from scipy import signal
from scipy.stats import zscore
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

class PerceptualLoss:
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()
        
    def normalize_for_vgg(self, x):
        """
        Normalizes a grayscale image tensor for VGG.
        Args:
            x (torch.Tensor): A tensor of shape (b, 1, 224, 224) with values in [0, 1].
        Returns:
            torch.Tensor: A tensor of shape (b, 3, 224, 224) normalized with ImageNet stats.
        """
        # Expand the grayscale image to 3 channels
        x = x.expand(-1, 3, -1, -1)
        
        # Define ImageNet mean and std and ensure they are on the same device as x
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        # Normalize the tensor
        x = (x - mean) / std
        return x

    def get_loss(self, fake, real):
        fake_norm = self.normalize_for_vgg(fake)
        real_norm = self.normalize_for_vgg(real)
        f_fake = self.contentFunc.forward(fake_norm)
        f_real = self.contentFunc.forward(real_norm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


def total_variation_loss_anisotropic(x):
    """
    Compute the total variation loss for a tensor x.
    Assumes x is of shape [B, C, H, W].
    """
    # Compute the differences along height and width
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    return tv_h.mean() + tv_w.mean()

def total_variation_loss_isotropic_masked(x, mask, eps=1e-8):
    """
    Isotropic TV loss with masking.

    Args:
        x    : [B, C, H, W] tensor
        mask : [B, 1, H, W] boolean or {0,1} float mask
    """
    mask = mask.float()

    # vertical and horizontal diffs
    diff_h = x[:, :, 1:, :-1] - x[:, :, :-1, :-1]   # [B, C, H-1, W-1]
    diff_w = x[:, :, :-1, 1:] - x[:, :, :-1, :-1]   # [B, C, H-1, W-1]

    # corresponding validity masks: both pixels must be inside mask
    valid_h = mask[:, :, 1:, :-1] * mask[:, :, :-1, :-1]
    valid_w = mask[:, :, :-1, 1:] * mask[:, :, :-1, :-1]

    # Euclidean norm of diffs
    tv_vals = torch.sqrt(diff_h**2 + diff_w**2 + eps)

    # combine masks (logical OR so we keep edges in either direction)
    valid = (valid_h + valid_w).clamp(max=1)

    # average only over valid pixels
    if valid.sum() > 0:
        loss = (tv_vals * valid).sum() / valid.sum()
    else:
        loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    return loss

def total_variation_loss_isotropic(x):
    """
    Compute the isotropic total variation loss for a tensor x.
    Assumes x is of shape [B, C, H, W].

    The isotropic total variation is computed as:
      TV(x) = mean_{i,j} sqrt( (x[:,:,i+1,j] - x[:,:,i,j])**2 + (x[:,:,i,j+1] - x[:,:,i,j])**2 )
    We compute the differences on the grid excluding the last row and column,
    so that both vertical and horizontal differences are computed on the same [B, C, H-1, W-1] region.
    """
    # Compute vertical differences: from pixel (i, j) to pixel (i+1, j)
    diff_h = x[:, :, 1:, :-1] - x[:, :, :-1, :-1]

    # Compute horizontal differences: from pixel (i, j) to pixel (i, j+1)
    diff_w = x[:, :, :-1, 1:] - x[:, :, :-1, :-1]

    # Add small epsilon to prevent numerical instability
    epsilon = 1e-8
    
    # Compute the isotropic total variation loss using the Euclidean norm of the differences
    loss = torch.sqrt(diff_h**2 + diff_w**2 + epsilon).mean()

    return loss

def compute_invariant_loss(fake, real, loss_fn):
    """
    Computes scale-invariant normalized loss between `fake` and `real` using `loss_fn`.
    Normalization is done using the norm of the real image.
    """
    norm_real = torch.norm(real, p=2, dim=(1, 2, 3), keepdim=True) + 1e-6
    normed_fake = fake / norm_real
    normed_real = real / norm_real
    return loss_fn(normed_fake, normed_real)

def compute_pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes a differentiable Pearson correlation loss between pred and target.

    Args:
        pred (torch.Tensor): Predicted tensor of shape (B, ...).
        target (torch.Tensor): Ground-truth tensor of same shape as pred.
        eps (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: Scalar tensor representing the mean Pearson loss over the batch,
                      defined as 1 - correlation, in [0, 2].
    """
    # Flatten spatial dimensions per sample: (B, N)
    batch_size = pred.shape[0]
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    # Subtract means
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean

    # Compute covariance and variances
    covariance = (pred_centered * target_centered).sum(dim=1)
    var_pred = (pred_centered ** 2).sum(dim=1)
    var_target = (target_centered ** 2).sum(dim=1)

    # Compute Pearson correlation per sample
    corr = covariance / (torch.sqrt(var_pred * var_target) + eps)

    # Loss is 1 - correlation; clamp to [0, 2] if desired
    loss = 1.0 - corr

    # Return mean loss over the batch
    return loss.mean()

def PSNR(img1, img2, mask=None):
    """Peak signal-to-noise ration of image (or within mask if defined)"""
    if mask is None:
        mse = torch.mean(torch.pow(img1 - img2, 2))
    else:
        mse = (img1[mask] - img2[mask]).pow(2).sum() / len(np.where(mask.cpu())[0])

    return 10 * torch.log10(1 / mse).clone().cpu().detach().numpy().item()


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def SSIM_crop(t1, t2, mask_t):
    """
    Return the Structural Similarity Map corresponding to input images img1 and img2

    This function attempts to mimic precisely the functionality of ssim.m a MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    ssim = 0
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K = 0.03
    L = 4095  # bitdepth of image
    C = (K * L) ** 2
    for i in range(t1.shape[0]):
        img1 = t1[i, 0, :, :].detach().cpu().numpy()*L
        img2 = t2[i, 0, :, :].detach().cpu().numpy()*L
        mask = mask_t[i, 0, :, :].detach().cpu().numpy()

        mu1 = signal.fftconvolve(window, img1, mode='valid')
        mu2 = signal.fftconvolve(window, img2, mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
        sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
        sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2

        ssim_map = (2.0 * sigma12 + C) / (sigma1_sq + sigma2_sq + C)
        ssim += np.sum(ssim_map[mask[5:-5, 5:-5]])/np.count_nonzero(mask)  # sum over batch

    return ssim/t1.shape[0]  # average of batch


def estimate_blur(T):
    """ Return estimation of blur using Laplacian (higher value means sharper) """
    L = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])
    blur = 0
    for i in range(T.shape[0]):
        I = T[i, 0, :, :].detach().cpu().numpy()*255

        blur += signal.convolve2d(I, L).var()

    return blur/T.shape[0]


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    # apply mask to SSIM map?

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def PD_constraint_loss(PD, mask, threshold=0.60):
    # Expand PD to include a channel dimension.
    PD = PD[:, None, :, :]
    
    # If no mask is provided or mask is all False, return 0 loss.
    if mask is None or not torch.any(mask):
        return torch.tensor(0.0, device=PD.device)
    # Select only the pixels that are within the mask.
    PD_masked = PD[mask]
    # Compute per-pixel hinge loss: for each pixel, loss = max(threshold - pixel, 0)
    hinge_loss = torch.clamp(threshold - PD_masked, min=0)
    # Aggregate the hinge losses
    return hinge_loss.mean()


def gaussian_prior_loss(pred: torch.Tensor, mask: torch.BoolTensor, mu: float, sigma: float) -> torch.Tensor:
    """
    Encourage pred[mask] to lie around mu with variance sigma^2.
    Loss = mean( (x - mu)^2 / (2 sigma^2) ).
    """
    # make sure mask has no extra singleton dims
    mask = mask.squeeze()
    # this will return a 1-D tensor of all pred[i] where mask[i] == True
    vals = torch.masked_select(pred, mask)
    if vals.numel() == 0:
        return torch.tensor(0., device=pred.device)
    return torch.mean((vals - mu) ** 2 / (2 * sigma ** 2))

