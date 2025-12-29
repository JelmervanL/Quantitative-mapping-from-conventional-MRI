import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchio as tio
import SimpleITK as sitk
from typing import Optional, List, Dict, Union
from qmap.util.util import (
    ModeNormalization, 
    OtsuMaskTransform, 
    RemoveSparseMaskedSlices, 
    CustomCropPad
)

def set_transforms(phase, thresholds_map=None):
    transforms = [tio.ToCanonical(), CustomCropPad()]
    
    if phase == 'train':
        transforms.append(RemoveSparseMaskedSlices(mask_name='brain_mask', min_voxels=5000))
        
    transforms.append(OtsuMaskTransform(t1_image_name='T1w', mask_name='otsu_mask'))
    
    if thresholds_map:
        transforms.append(
            ModeNormalization(
                masking_method='otsu_mask', # Use the computed otsu mask
                thresholds_map=thresholds_map
            )
        )

    return tio.Compose(transforms)

# --- DATASET CREATION HELPERS ---
def _adjust_te_params(df, root):
    """Corrects TE parameters based on dataset heuristics."""
    df['FLAIR_TE'] = df['FLAIR_TE'].apply(lambda x: x * 0.89 if x < 200 else x * 0.42)
    df['T2w_TE'] = df['T2w_TE'] * 0.9
        
    if 'FLAIR_TE' not in df.columns: raise ValueError("FLAIR_TE missing from CSV")
    if 'T2w_TE' not in df.columns: raise ValueError("T2w_TE missing from CSV")
    return df

def _load_subject_df(root, suffix, phase):
    path = os.path.join(root, phase + suffix)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

def _get_subject_list(df, opt, root):
    """Filters subject list based on subset/eval size options."""
    all_ids = df['ID'].values.tolist()
    
    # Training Subset
    if opt.phase == 'train' and opt.train_subset_size > 0 and opt.train_subset_size < len(all_ids):
        csv_names_tuple = tuple(opt.data_csv_name)
        seed_components = (root, csv_names_tuple, opt.random_seed)
        seed = abs(hash(seed_components)) % (2**32)
        rng = np.random.default_rng(seed)
        selected_ids = rng.choice(all_ids, size=opt.train_subset_size, replace=False).tolist()
        print(f"[Subset] {root}: Using {len(selected_ids)}/{len(all_ids)} subjects.")
        return selected_ids
    
    # Validation/Test Truncation
    if opt.phase in ('val', 'test') and opt.eval_size < len(all_ids):
        print(f"Selecting first {opt.eval_size} subjects.")
        return all_ids[:opt.eval_size]
        
    return all_ids

def _oversample_minority(subject_ids, df, opt):
    """Oversamples minority sequence types for balance."""
    target_col = "T1w_seq_type"
    target_val = "SE"
    
    df_sub = df[df['ID'].isin(subject_ids)]
    minority_ids = df_sub[df_sub[target_col] == target_val]['ID'].tolist()
    
    n_maj = len(subject_ids) - len(minority_ids)
    if not minority_ids or n_maj == 0: return subject_ids
    
    needed = math.ceil(0.1 * n_maj) - len(minority_ids)
    if needed <= 0: return subject_ids
    
    rng = np.random.default_rng(opt.random_seed)
    extras = rng.choice(minority_ids, size=needed, replace=True).tolist()
    
    print(f"Oversampled {len(extras)} {target_val} subjects.")
    return subject_ids + extras

def _create_subject(row, root, opt):
    """Builds a single TorchIO Subject from a DataFrame row."""
    sub_id = row['ID']
    dir_img = os.path.join(root, 'images')
    dir_msk = os.path.join(root, 'masks')
    
    # Determine filenames
    fnames = {
        'T1w': sub_id + '_T1w.nii.gz',
        'T2w': sub_id + '_T2w.nii.gz',
        'FLAIR': sub_id + '_FLAIR.nii.gz'
    } if 'UMCU' in root else { # Default naming assumption
        'T1w': sub_id + '_T1w.nii.gz',
        'T2w': sub_id + '_T2w.nii.gz',
        'FLAIR': sub_id + '_FLAIR.nii.gz'
    }

    # Base Subject Dictionary
    kwargs = {
        'subject_id': sub_id,
        'brain_mask': tio.LabelMap(os.path.join(dir_msk, sub_id + '_brain.nii.gz')),
        
        # Sequence Params
        'T1w_TR': row['T1w_TR'], 'T1w_TE': row['T1w_TE'], 'T1w_FA': row['T1w_FA'], #'T1w_TI': row['T1w_TI'],
        'T2w_TR': row['T2w_TR'], 'T2w_TE': row['T2w_TE'], 'T2w_FA': row['T2w_FA'],
        'FLAIR_TR': row['FLAIR_TR'], 'FLAIR_TE': row['FLAIR_TE'], 'FLAIR_TI': row['FLAIR_TI'], 'FLAIR_FA': row['FLAIR_FA'],
        
        # Rescaling Factors
        'T1w_rescaling_factor': row['T1w_rescale'],
        'T2w_rescaling_factor': row['T2w_rescale'],
        'FLAIR_rescaling_factor': row['FLAIR_rescale'],
        
        # Sequence Types (with defaults)
        'T1w_seq_type': row.get('T1w_seq_type', 'TFE'),
        'T2w_seq_type': row.get('T2w_seq_type', 'TSE'),
        'FLAIR_seq_type': row.get('FLAIR_seq_type', 'IR'),
    }

    # Load Images
    for mod, fname in fnames.items():
        path = os.path.join(dir_img, fname)
        if not os.path.exists(path): raise FileNotFoundError(f"Missing {mod}: {path}")
        kwargs[mod] = tio.ScalarImage(path)

    # Load Tissue Masks (if needed)
    need_masks = opt.eval_qstar_tissue_values or opt.loss_PD_prior or opt.rescaling_method == 'mean'
    if need_masks:
        for tissue in ['csf', 'gm', 'wm']:
            p = os.path.join(dir_msk, f"{sub_id}_{tissue}.nii.gz")
            if os.path.exists(p):
                kwargs[f"{tissue}_mask"] = tio.LabelMap(p)

    return tio.Subject(**kwargs)

# --- CREATE DATASET FUNCTION ---
def create_dataset(opt):
    print("Creating dataset...")
    
    roots = opt.dataroot if isinstance(opt.dataroot, list) else [opt.dataroot]
    suffixes = opt.data_csv_name if isinstance(opt.data_csv_name, list) else [opt.data_csv_name]
    
    if len(roots) != len(suffixes):
        raise ValueError("Roots and CSV suffixes lists must be equal length.")
        
    all_subjects = []
    
    for root, suffix in zip(roots, suffixes):
        df = _load_subject_df(root, suffix, opt.phase)
        df = _adjust_te_params(df, root)
        
        subject_ids = _get_subject_list(df, opt, root)
        
        if opt.phase == "train" and opt.isTrain and getattr(opt, "oversample_T1w_SE", False):
            subject_ids = _oversample_minority(subject_ids, df, opt)
            
        print(f"Loading {len(subject_ids)} subjects from {root}...")
        
        # Build subject objects
        for sub_id in subject_ids:
            row = df[df['ID'] == sub_id].iloc[0]
            try:
                all_subjects.append(_create_subject(row, root, opt))
            except FileNotFoundError as e:
                print(f"Skipping {sub_id}: {e}")

    # Transforms
    thresholds = {
        'T1w': {'lower': 60, 'upper': 95},
        'T2w': {'lower': 35, 'upper': 95},
        'FLAIR': {'lower': 35, 'upper': 95},
    }
    transform = set_transforms(opt.phase, thresholds)
    
    return tio.SubjectsDataset(all_subjects, transform=transform)