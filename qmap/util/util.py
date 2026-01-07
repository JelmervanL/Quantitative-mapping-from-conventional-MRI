from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torchio as tio
import torch.nn.functional as F
import SimpleITK as sitk
import torch.nn as nn
import random
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def calc_qstar_stats(PD, T1, T2, m_wm, m_gm, m_csf):
    """Helper to calculate means for a single subject."""
    return {
        'PD_WM': PD[m_wm].mean(), 'PD_GM': PD[m_gm].mean(), 'PD_CSF': PD[m_csf].mean(),
        'T1_WM': T1[m_wm].mean(), 'T1_GM': T1[m_gm].mean(), 'T1_CSF': T1[m_csf].mean(),
        'T2_WM': T2[m_wm].mean(), 'T2_GM': T2[m_gm].mean(), 'T2_CSF': T2[m_csf].mean(),
    }
    
def _parse_int_list(s: str):
    return [int(x) for x in s.split(',') if x.strip()]

def save_nifti(tensor, affine, path):
    """Helper to save torch tensor/numpy array as NIfTI."""
    if isinstance(tensor, torch.Tensor):
        data = tensor.numpy()
    else:
        data = tensor
    data = data.squeeze()
    # Check for empty mask saving boolean arrays as floats for NIfTI compatibility
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)

class ModeNormalization(tio.transforms.preprocessing.intensity.NormalizationTransform):
    """Normalize image intensity by mode using a histogram-based approach."""
    def __init__(self, masking_method, thresholds_map=None, **kwargs):
        super().__init__(masking_method=masking_method, **kwargs)
        self.args_names = ['masking_method', 'thresholds_map']
        self.thresholds_map = thresholds_map or {}

    def apply_normalization(self, subject, image_name, mask):
        image = subject[image_name]
        data = image.data
        
        thresholds = self.thresholds_map.get(image_name, {})
        normalized = self.modenorm(
            data, mask,
            percentile_threshold_lower=thresholds.get('lower'),
            percentile_threshold_upper=thresholds.get('upper')
        )

        if normalized is None:
            if torch.max(data) == 0:
                return subject
            raise RuntimeError(f'Mode is 0 for masked values in image "{image_name}"')
        
        image.set_data(normalized)

    @staticmethod
    def get_mode(voxels, bins=512, lower_pct=None, upper_pct=None):
        if lower_pct:
            voxels = voxels[voxels >= torch.quantile(voxels, lower_pct / 100.0)]
        if upper_pct:
            voxels = voxels[voxels <= torch.quantile(voxels, upper_pct / 100.0)]

        if voxels.numel() == 0: return None

        min_val, max_val = torch.min(voxels), torch.max(voxels)
        hist = torch.histc(voxels, bins=bins, min=min_val.item(), max=max_val.item())
        bin_edges = torch.linspace(min_val, max_val, steps=bins+1)
        mode_val = 0.5 * (bin_edges[torch.argmax(hist)] + bin_edges[torch.argmax(hist) + 1])
        return mode_val.item()

    @staticmethod
    def modenorm(tensor, mask, percentile_threshold_lower=None, percentile_threshold_upper=None):
        tensor = tensor.clone().float()
        mode_val = ModeNormalization.get_mode(
            tensor[mask], 
            lower_pct=percentile_threshold_lower, 
            upper_pct=percentile_threshold_upper
        )
        return (tensor / mode_val) if mode_val and mode_val > 0 else None

class OtsuMaskTransform(tio.Transform):
    """Compute a brain mask from T1w using Otsu and attach it to the subject."""
    def __init__(self, t1_image_name='T1w', mask_name='otsu_mask'):
        super().__init__()
        self.t1_name = t1_image_name
        self.mask_name = mask_name

    def apply_transform(self, subject):
        t1 = subject[self.t1_name]
        
        # Convert Torch -> SimpleITK
        sitk_img = sitk.GetImageFromArray(t1.data.numpy().squeeze(0))
        sitk_img.SetDirection(t1.affine[:3, :3].flatten().tolist())
        sitk_img.SetOrigin(tuple(t1.affine[:3, 3].tolist()))

        # Otsu Thresholding
        sitk_mask = sitk.OtsuThreshold(sitk.InvertIntensity(sitk.Cast(sitk_img, sitk.sitkFloat32)))

        # Convert back to Torch
        mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask).astype(np.uint8)).unsqueeze(0)
        
        subject.add_image(
            image=tio.LabelMap(tensor=mask_tensor, affine=t1.affine),
            image_name=self.mask_name
        )
        return subject

class RemoveSparseMaskedSlices(tio.Transform):
    """Drop depth slices with too few masked voxels."""
    def __init__(self, mask_name='brain_mask', min_voxels=25):
        super().__init__()
        self.mask_name = mask_name
        self.min_voxels = min_voxels

    def apply_transform(self, subject):
        mask = subject[self.mask_name].data
        if mask.ndimension() == 4: mask = mask.squeeze(0)
        
        valid_slices = mask.sum(dim=(0, 1)) >= self.min_voxels
        
        if not valid_slices.any():
            raise RuntimeError(f"No valid slices for subject {subject.get('subject_id')}")
        
        # --- Added Warning Logic ---
        original_count = mask.shape[-1]
        kept_count = valid_slices.sum().item()
        removed_count = original_count - kept_count

        if removed_count > 0:
            print(f"Warning: Removed {removed_count} slices from subject {subject.get('subject_id', 'unknown')}.")

        for image in subject.get_images(intensity_only=False):
            image.set_data(image.data[..., valid_slices])
            
        return subject

class CustomCropPad(tio.Transform):
    """Crops depth slices and pads/crops in-plane to 224x224."""
    def __init__(self, top_crop=0.05, bottom_crop=0.05):
        super().__init__()
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop

    def apply_transform(self, subject):
        for image in subject.get_images(intensity_only=False):
            data = image.data # (C, H, W, D)
            C, _, _, D = data.shape
            
            # Depth Cropping
            n_top = int(self.top_crop * D) if isinstance(self.top_crop, float) else self.top_crop
            n_bottom = int(self.bottom_crop * D) if isinstance(self.bottom_crop, float) else self.bottom_crop
            
            if D - n_top - n_bottom <= 0:
                 raise RuntimeError(f"Invalid crop amounts: {n_top} + {n_bottom} >= {D}")
                 
            data = data[..., n_top : D - n_bottom]
            
            # In-plane Resizing (224x224)
            processed_slices = []
            target_sz = 224
            
            for i in range(data.shape[-1]):
                sl = data[..., i] # (C, H, W)
                
                # Height Processing
                h = sl.shape[1]
                if h > target_sz:
                    start = (h - target_sz) // 2
                    sl = sl[:, start:start+target_sz, :]
                elif h < target_sz:
                    diff = target_sz - h
                    sl = F.pad(sl, (0, 0, diff//2, diff - diff//2))
                
                # Width Processing
                w = sl.shape[2]
                if w > target_sz:
                    start = (w - target_sz) // 2
                    sl = sl[:, :, start:start+target_sz]
                elif w < target_sz:
                    diff = target_sz - w
                    sl = F.pad(sl, (diff//2, diff - diff//2, 0, 0))
                    
                processed_slices.append(sl.unsqueeze(-1))

            image.set_data(torch.cat(processed_slices, dim=-1))
            
        return subject
    
class SliceVisualizer:
    def __init__(self, script_dir):
        self.script_dir = script_dir

    def color_log_remap(self, ori_cmap, loLev, upLev):
        """Re-map the original colormap along a “log-like” curve."""
        assert upLev > 0, "upper level must be positive"
        assert upLev > loLev, "upper level must be larger than lower level"
        
        map_length = ori_cmap.shape[0]
        e_inv = np.exp(-1.0)
        aVal = e_inv * upLev
        mVal = max(aVal, loLev)
        if aVal >= loLev:
            bVal = (1.0 / map_length) + ((aVal - loLev) / (2 * aVal - loLev))
        else:
            bVal = 1.0 / map_length
        bVal += 1e-7  # avoid rounding issues
        
        log_cmap = np.zeros_like(ori_cmap)
        log_cmap[0, :] = ori_cmap[0, :]
        log_portion = 1.0 / (np.log(mVal) - np.log(upLev))
        
        for g in range(1, map_length):
            x = (g + 1) * (upLev - loLev) / map_length + loLev
            f = 0.0
            if x > mVal:
                f = map_length * ((np.log(mVal) - np.log(x)) * log_portion * (1 - bVal) + bVal)
            else:
                if (loLev < aVal) and (x > loLev):
                    f = map_length * (((x - loLev) / (aVal - loLev)) * (bVal - (1.0 / map_length))) + 1.0
                if x <= loLev:
                    f = 1.0
            idx = min(map_length, 1 + int(np.floor(f))) - 1
            log_cmap[g, :] = ori_cmap[idx, :]
        
        return log_cmap

    def relaxation_color_map(self, maptype, x, loLev, upLev):
        """Read CSV colormap, clip image, return clipped image and custom cmap."""
        mmm = maptype[0].upper() + maptype[1:]
        
        if mmm in ['T1', 'R1']:
            csv_filename = os.path.join(self.script_dir, 'lipari.csv') # 
        elif mmm in ['T2', 'T2*', 'R2', 'R2*', 'T1rho', 'T1ρ', 'R1rho', 'R1ρ']:
            csv_filename = os.path.join(self.script_dir, 'navia.csv') # 
        else:
            raise ValueError("Expect 'T1', 'T2', 'R1', or 'R2' as maptype")
        
        if not os.path.exists(csv_filename):
            print(f"Warning: Colormap file {csv_filename} not found. Using default viridis.")
            return x, 'viridis'

        colortable = np.loadtxt(csv_filename, delimiter=' ', skiprows=1)
        if mmm[0] == 'R':
            colortable = np.flipud(colortable)
        colortable[0, :] = 0.0
        map_length = colortable.shape[0]
        eps_val = (upLev - loLev) / map_length
        
        x_clip = np.where(x < eps_val,
                        np.where(x < (loLev + eps_val), loLev - eps_val, loLev + eps_val),
                        x)
        if loLev < 0:
            x_clip = np.where(x < eps_val, loLev - eps_val, x)
        
        lut_cmap = self.color_log_remap(colortable, loLev, upLev)
        return x_clip, lut_cmap

    def save_subject_figure(self, save_path, real_imgs, fake_imgs, q_maps, subject_id):
        """
        Generates and saves the 3x3 grid figure.
        real_imgs: [T1w, T2w, FLAIR]
        fake_imgs: [Fake T1w, Fake T2w, Fake FLAIR]
        q_maps: [PD, T1, T2]
        """
        # Unpack
        r_t1, r_t2, r_flair = real_imgs
        f_t1, f_t2, f_flair = fake_imgs
        q_pd, q_t1, q_t2 = q_maps

        # Get center slice index along z-axis
        sl = r_t1.shape[2] // 2
        
        # --- HELPER: Extract, Rotate, Flip, and CROP Slice ---
        def get_sl(vol): 
            # 1. Extract axial slice, rotate 90 deg, flip left-right
            img_2d = np.fliplr(np.rot90(vol[:, :, sl]))
            # 2. Crop 25 pixels from left and right only
            return img_2d[:, 32:-32]

        # --- Setup Compact Figure ---
        fig, axs = plt.subplots(3, 3, figsize=(8, 10), facecolor='black', constrained_layout=True,
                                gridspec_kw={'wspace': 0.0, 'hspace': 0.0})
        
        # Define subplot helper
        def plot_ax(ax, img, title, cmap='gray', vmin=None, vmax=None, show_cbar=False):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, color='white', fontsize=10, pad=5)
            ax.axis('off')
            if show_cbar:
                # Place colorbar at bottom to save horizontal space
                cbar = plt.colorbar(im, ax=ax, location='bottom', shrink=1, aspect=30, pad=0.02)
                cbar.ax.xaxis.set_tick_params(color='white', labelcolor='white', labelsize=8)
                cbar.outline.set_edgecolor('white')

        # --- Row 1: Real Inputs (Cropped) ---
        plot_ax(axs[0, 0], get_sl(r_t1), 'Real T1w')
        plot_ax(axs[0, 1], get_sl(r_t2), 'Real T2w')
        plot_ax(axs[0, 2], get_sl(r_flair), 'Real FLAIR')

        # --- Row 2: Synthesized (Cropped) ---
        plot_ax(axs[1, 0], get_sl(f_t1), 'Synth T1w')
        plot_ax(axs[1, 1], get_sl(f_t2), 'Synth T2w')
        plot_ax(axs[1, 2], get_sl(f_flair), 'Synth FLAIR')

        # --- Row 3: Q-Maps (Cropped + Custom Colormaps) ---
        
        # PD (Gray, 0-1)
        plot_ax(axs[2, 0], get_sl(q_pd), 'Quantitative PD', cmap='gray', vmin=0, vmax=1, show_cbar=True)

        # T1 (Custom)
        t1_slice = get_sl(q_t1)
        T1_lo, T1_up = 300, 3000
        t1_clip, lut_t1 = self.relaxation_color_map('T1', t1_slice, T1_lo, T1_up)
        if isinstance(lut_t1, str): cmap_t1 = lut_t1 
        else: cmap_t1 = matplotlib.colors.ListedColormap(lut_t1)
        
        plot_ax(axs[2, 1], t1_clip, 'Quantitative T1 (ms)', cmap=cmap_t1, vmin=T1_lo, vmax=T1_up, show_cbar=True)

        # T2 (Custom)
        t2_slice = get_sl(q_t2)
        T2_lo, T2_up = 40, 200
        t2_clip, lut_t2 = self.relaxation_color_map('T2', t2_slice, T2_lo, T2_up)
        if isinstance(lut_t2, str): cmap_t2 = lut_t2
        else: cmap_t2 = matplotlib.colors.ListedColormap(lut_t2)

        plot_ax(axs[2, 2], t2_clip, 'Quantitative T2 (ms)', cmap=cmap_t2, vmin=T2_lo, vmax=T2_up, show_cbar=True)

        # Save
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
# images not normalized properly for visualization?
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] > 3:
        image_numpy = np.tile(image_numpy[0, :, :], (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    if image_numpy.shape[0] == 3:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if image_numpy.shape[0] == 2:
        img = np.sqrt(np.square(image_numpy[1, :, :])+np.square(image_numpy[0, :, :]))
        upp = np.percentile(img, 99.9)
        lowp = np.percentile(img, 0.1)
        img = np.clip(img, a_min=lowp, a_max=upp)
        image_numpy = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        image_numpy = image_numpy/(np.amax(image_numpy)+0.0000001)*255.0

    return image_numpy.astype(imtype)


def tensor2imdiff(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))+1)/2 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_phase(batch_input):
    temp = batch_input[:, 2, :, :]/batch_input[:, 1, :, :]
    return temp.unsqueeze(1)


def get_amp(batch_input):
    return torch.sqrt(torch.pow(batch_input[:, 2, :, :], 2) + torch.pow(batch_input[:, 1, :, :], 2))


def complex_matmul(a, b):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the last two channels.
    return a[..., 0]*b[..., 0]-a[..., 1]*b[..., 1]
