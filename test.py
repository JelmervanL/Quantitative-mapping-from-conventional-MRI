# #!/usr/bin/env python3
# """
# test.py

# This script loads a saved model (from a specified checkpoint) and performs testing/inference on the test dataset.
# It uses a grid sampler/aggregator (via TorchIO) to run inference patch‐wise over each test subject,
# reconstructs the full output images, computes metrics (e.g. SSIM, PSNR, L1, etc.), and calculates qstar values.
# Results (metrics and optional NIfTI outputs) are saved in the test_results folder.
# """

# import os
# import time
# import numpy as np
# import pandas as pd
# import nibabel as nib
# import torch
# import torchio as tio
# from tqdm import tqdm
# from scipy.ndimage import binary_erosion
# import matplotlib
# matplotlib.use('Agg')  # for non-interactive backend
# from typing import Dict, Tuple

# from Qstar.options.test_options import TestOptions
# from Qstar.models import create_model
# from Qstar.data.conventional_dataset import create_dataset, create_dataset_multiprot_cond, create_dataset_volunteer
# from Qstar.util.visualizer import Visualizer

# def calc_values_qstarmaps(
#     all_PD,
#     all_T1,
#     all_T2,
#     all_mask_wm,
#     all_mask_gm,
#     all_mask_csf,
#     all_mask_tumor=None,
#     cutoffs: Dict[str, Tuple[float, float]] = None
# ):
#     """
#     Compute mean & std of PD / T1 / T2 for WM, GM, CSF (and tumour sub-types),
#     discarding values outside tissue-specific lower/upper percentile bounds.

#     Parameters
#     ----------
#     cutoffs : dict
#         e.g. {"wm": (1, 99), "gm": (2, 98), "csf": (0.5, 99.5)}
#         Percentiles are inclusive bounds *to keep* (low, high).
#         Missing keys default to (0, 100). Tumour uses cutoffs["tumor"] if present.
#     """
#     # ---------------- helpers -------------------------------------------------
#     def remove_outliers(data, low, high):
#         if data.size == 0:
#             return data
#         lo_val, hi_val = np.percentile(data, [low, high])
#         return data[(data >= lo_val) & (data <= hi_val)]

#     # set defaults
#     if cutoffs is None:
#         cutoffs = {}
#     wm_low, wm_high = cutoffs.get("wm", (0, 97))
#     gm_low, gm_high = cutoffs.get("gm", (2, 99))
#     csf_low, csf_high = cutoffs.get("csf", (5, 100))
#     tumor_low, tumor_high = cutoffs.get("tumor", (0, 100))

#     # ---------------- flatten & mask -----------------------------------------
#     PD, T1, T2 = (arr.flatten().astype(np.float64) for arr in (all_PD, all_T1, all_T2))
#     mask_wm, mask_gm, mask_csf = (m.flatten().astype(bool) for m in (all_mask_wm,
#                                                                      all_mask_gm,
#                                                                      all_mask_csf))

#     # ---------------- tissue-specific values ---------------------------------
#     PD_wm = remove_outliers(PD[mask_wm], wm_low, wm_high)
#     PD_gm = remove_outliers(PD[mask_gm], gm_low, gm_high)
#     PD_csf = remove_outliers(PD[mask_csf], csf_low, csf_high)

#     T1_wm = remove_outliers(T1[mask_wm], wm_low, wm_high)
#     T1_gm = remove_outliers(T1[mask_gm], gm_low, gm_high)
#     T1_csf = remove_outliers(T1[mask_csf], csf_low, csf_high)

#     T2_wm = remove_outliers(T2[mask_wm], wm_low, wm_high)
#     T2_gm = remove_outliers(T2[mask_gm], gm_low, gm_high)
#     T2_csf = remove_outliers(T2[mask_csf], csf_low, csf_high)

#     # ---------------- stats ---------------------------------------------------
#     PD_wm_mean, PD_gm_mean, PD_csf_mean = map(np.mean, (PD_wm, PD_gm, PD_csf))
#     T1_wm_mean, T1_gm_mean, T1_csf_mean = map(np.mean, (T1_wm, T1_gm, T1_csf))
#     T2_wm_mean, T2_gm_mean, T2_csf_mean = map(np.mean, (T2_wm, T2_gm, T2_csf))

#     PD_wm_std, PD_gm_std, PD_csf_std = map(np.std, (PD_wm, PD_gm, PD_csf))
#     T1_wm_std, T1_gm_std, T1_csf_std = map(np.std, (T1_wm, T1_gm, T1_csf))
#     T2_wm_std, T2_gm_std, T2_csf_std = map(np.std, (T2_wm, T2_gm, T2_csf))

#     # ---------------- tumour sub-types (optional) ----------------------------
#     # initialise with Nones so return signature unchanged
#     PD_necrosis_mean = PD_enhancing_tumor_mean = PD_invasion_mean = None
#     T1_necrosis_mean = T1_enhancing_tumor_mean = T1_invasion_mean = None
#     T2_necrosis_mean = T2_enhancing_tumor_mean = T2_invasion_mean = None
#     PD_necrosis_std = PD_enhancing_tumor_std = PD_invasion_std = None
#     T1_necrosis_std = T1_enhancing_tumor_std = T1_invasion_std = None
#     T2_necrosis_std = T2_enhancing_tumor_std = T2_invasion_std = None

#     if all_mask_tumor is not None:
#         tumor_mask = all_mask_tumor.flatten().astype(int)
#         sub_masks = {
#             "necrosis": tumor_mask == 1,
#             "enhancing_tumor": tumor_mask == 2,
#             "invasion": tumor_mask == 4,
#         }

#         def proc(arr, m):
#             return remove_outliers(arr[m], tumor_low, tumor_high)

#         PD_necrosis, PD_enhancing_tumor, PD_invasion = [proc(PD, m) for m in sub_masks.values()]
#         T1_necrosis, T1_enhancing_tumor, T1_invasion = [proc(T1, m) for m in sub_masks.values()]
#         T2_necrosis, T2_enhancing_tumor, T2_invasion = [proc(T2, m) for m in sub_masks.values()]

#         PD_necrosis_mean, PD_enhancing_tumor_mean, PD_invasion_mean = (
#             np.mean(PD_necrosis) if PD_necrosis.size else None,
#             np.mean(PD_enhancing_tumor) if PD_enhancing_tumor.size else None,
#             np.mean(PD_invasion) if PD_invasion.size else None,
#         )
#         T1_necrosis_mean, T1_enhancing_tumor_mean, T1_invasion_mean = (
#             np.mean(T1_necrosis) if T1_necrosis.size else None,
#             np.mean(T1_enhancing_tumor) if T1_enhancing_tumor.size else None,
#             np.mean(T1_invasion) if T1_invasion.size else None,
#         )
#         T2_necrosis_mean, T2_enhancing_tumor_mean, T2_invasion_mean = (
#             np.mean(T2_necrosis) if T2_necrosis.size else None,
#             np.mean(T2_enhancing_tumor) if T2_enhancing_tumor.size else None,
#             np.mean(T2_invasion) if T2_invasion.size else None,
#         )

#         PD_necrosis_std, PD_enhancing_tumor_std, PD_invasion_std = (
#             np.std(PD_necrosis) if PD_necrosis.size else None,
#             np.std(PD_enhancing_tumor) if PD_enhancing_tumor.size else None,
#             np.std(PD_invasion) if PD_invasion.size else None,
#         )
#         T1_necrosis_std, T1_enhancing_tumor_std, T1_invasion_std = (
#             np.std(T1_necrosis) if T1_necrosis.size else None,
#             np.std(T1_enhancing_tumor) if T1_enhancing_tumor.size else None,
#             np.std(T1_invasion) if T1_invasion.size else None,
#         )
#         T2_necrosis_std, T2_enhancing_tumor_std, T2_invasion_std = (
#             np.std(T2_necrosis) if T2_necrosis.size else None,
#             np.std(T2_enhancing_tumor) if T2_enhancing_tumor.size else None,
#             np.std(T2_invasion) if T2_invasion.size else None,
#         )

#     # ---------------- return --------------------------------------------------
#     return (
#         PD_wm_mean, PD_gm_mean, PD_csf_mean,
#         PD_wm_std,  PD_gm_std,  PD_csf_std,
#         PD_necrosis_mean, PD_enhancing_tumor_mean, PD_invasion_mean,
#         PD_necrosis_std,  PD_enhancing_tumor_std,  PD_invasion_std,

#         T1_wm_mean, T1_gm_mean, T1_csf_mean,
#         T1_wm_std,  T1_gm_std,  T1_csf_std,
#         T1_necrosis_mean, T1_enhancing_tumor_mean, T1_invasion_mean,
#         T1_necrosis_std,  T1_enhancing_tumor_std,  T1_invasion_std,

#         T2_wm_mean, T2_gm_mean, T2_csf_mean,
#         T2_wm_std,  T2_gm_std,  T2_csf_std,
#         T2_necrosis_mean, T2_enhancing_tumor_mean, T2_invasion_mean,
#         T2_necrosis_std,  T2_enhancing_tumor_std,  T2_invasion_std
#     )

# def main():
#     # Parse test options
#     opt = TestOptions().parse()
#     opt.phase = 'test'
#     print("Running in test phase:", opt.phase)

#     # Create the test dataset. Choose the multi-protocol variant if needed.
#     if opt.multi_protocol_conditioning:
#         dataset_test = create_dataset_multiprot_cond(opt)
#     elif opt.use_volunteer_dataset:
#         dataset_test = create_dataset_volunteer(opt)
#     else:
#         dataset_test = create_dataset(opt)
#     dataset_size = len(dataset_test)
#     print("# Test subjects:", dataset_size)

#     # Create directory to save test results
#     web_dir = os.path.join(opt.checkpoints_dir, opt.name)
#     test_results_dir = os.path.join(web_dir, opt.results_dir)
#     os.makedirs(test_results_dir, exist_ok=True)

#     # Create the model and load the saved weights
#     model = create_model(opt)
#     model.setup(opt)
#     print("Loading model weights from epoch:", opt.which_epoch_load)
#     model.load_networks(opt.which_epoch_load)

#     # Instantiate the visualizer (if you want to save example images)
#     visualizer = Visualizer(opt)

#     # Define the patch size (should match what was used in training)
#     patch_size = (224, 224, 1)
#     batch_size = opt.batchSize

#     # DataFrames for aggregating per-subject metrics and qstar values
#     test_metrics = pd.DataFrame(columns=[
#        'subject_id',
#        'SSIM_T1w', 'SSIM_T2w', 'SSIM_FLAIR',
#        'PSNR_T1w', 'PSNR_T2w', 'PSNR_FLAIR',
#        'L1_overall', 'L2_overall',
#        'L1_T1w', 'L1_T2w', 'L1_FLAIR',
#        'L2_T1w', 'L2_T2w', 'L2_FLAIR',
#        'vgg', 'PDT1_relation', 'PD_wm', 'PD_variance', 'PD_constraint_head', 'tv_reg',
#        'rescaling_factor_T1w', 'rescaling_factor_T2w', 'rescaling_factor_FLAIR'
#     ])
#     test_qstar_values = pd.DataFrame(columns=[
#         'subject_id', 'protocol_id',
#         'PD_wm_mean', 'PD_gm_mean', 'PD_csf_mean',
#         'PD_wm_std', 'PD_gm_std', 'PD_csf_std',
#         'PD_necrosis_mean', 'PD_enhancing_tumor_mean', 'PD_invasion_mean',
#         'PD_necrosis_std', 'PD_enhancing_tumor_std', 'PD_invasion_std',
#         'T1_wm_mean', 'T1_gm_mean', 'T1_csf_mean',
#         'T1_wm_std', 'T1_gm_std', 'T1_csf_std',
#         'T1_necrosis_mean', 'T1_enhancing_tumor_mean', 'T1_invasion_mean',
#         'T1_necrosis_std', 'T1_enhancing_tumor_std', 'T1_invasion_std',
#         'T2_wm_mean', 'T2_gm_mean', 'T2_csf_mean',
#         'T2_wm_std', 'T2_gm_std', 'T2_csf_std',
#         'T2_necrosis_mean', 'T2_enhancing_tumor_mean', 'T2_invasion_mean',
#         'T2_necrosis_std', 'T2_enhancing_tumor_std', 'T2_invasion_std'
#     ])
    
#     if opt.aggregated_qstar_histograms:
#         aggregated_T1_wm = []
#         aggregated_T1_gm = []
#         aggregated_T2_wm = []
#         aggregated_T2_gm = []

#     # Loop over test subjects
#     for subject in tqdm(dataset_test, desc="Testing subjects"):
#         # Create a grid sampler and aggregators for patch‐wise inference
#         grid_sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap=(0, 0, 0), padding_mode=None)
#         aggregator_Q1 = tio.inference.GridAggregator(grid_sampler, "crop")
#         aggregator_Q2 = tio.inference.GridAggregator(grid_sampler, "crop")
#         aggregator_Q3 = tio.inference.GridAggregator(grid_sampler, "crop")
#         aggregator_T1w_fake = tio.inference.GridAggregator(grid_sampler, "crop")
#         aggregator_T2w_fake = tio.inference.GridAggregator(grid_sampler, "crop")
#         aggregator_FLAIR_fake = tio.inference.GridAggregator(grid_sampler, "crop")
        

        
#         loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size, num_workers=4)

#         # Initialize lists to collect metrics for this subject
#         L1_list_test = []
#         L2_list_test = []
#         L1_T1w_test, L1_T2w_test, L1_FLAIR_test = [], [], []
#         L2_T1w_test, L2_T2w_test, L2_FLAIR_test = [], [], []
#         SSIM_T1w_test, SSIM_T2w_test, SSIM_FLAIR_test = [], [], []
#         PSNR_T1w_test, PSNR_T2w_test, PSNR_FLAIR_test = [], [], []
#         vgg_test = []
#         PDT1_relation_test = []
#         PD_wm_test = []
#         PD_variance_test = []
#         rescaling_factor_T1w_test, rescaling_factor_T2w_test, rescaling_factor_FLAIR_test = [], [], []

#         # For qstar computation, we also accumulate all patch outputs
#         all_PD = None
#         all_T1 = None
#         all_T2 = None

#         # Inference on patches (with no gradient calculation)
#         with torch.no_grad():
#             for patch in loader:
#                 model.set_input(patch, phase_input='test')
#                 model.test()

#                 # Get patch locations and add outputs to aggregators.
#                 locations = patch[tio.LOCATION]
#                 aggregator_Q1.add_batch(model.Q1.view(model.Q1.shape[0], 1, *patch_size).detach().cpu(), locations)
#                 aggregator_Q2.add_batch(model.Q2.view(model.Q2.shape[0], 1, *patch_size).detach().cpu(), locations)
#                 aggregator_Q3.add_batch(model.Q3.view(model.Q3.shape[0], 1, *patch_size).detach().cpu(), locations)
#                 aggregator_T1w_fake.add_batch((model.fake_T1w * model.mask_brain).view(model.fake_T1w.shape[0], 1, *patch_size).detach().cpu(), locations)
#                 aggregator_T2w_fake.add_batch((model.fake_T2w * model.mask_brain).view(model.fake_T2w.shape[0], 1, *patch_size).detach().cpu(), locations)
#                 aggregator_FLAIR_fake.add_batch((model.fake_FLAIR * model.mask_brain).view(model.fake_FLAIR.shape[0], 1, *patch_size).detach().cpu(), locations)
                
               
#                 # Retrieve the losses/metrics from the model.
#                 losses = model.get_current_losses()

#                 if 'G_I_L1' in losses:
#                     L1_list_test.append(losses['G_I_L1'])
#                 if 'G_I_L2' in losses:
#                     L2_list_test.append(losses['G_I_L2'])
#                 # Record per‐modality L1 and L2 losses
#                 if 'G_I_L1_T1w' in losses:
#                     L1_T1w_test.append(losses['G_I_L1_T1w'])
#                 if 'G_I_L1_T2w' in losses:
#                     L1_T2w_test.append(losses['G_I_L1_T2w'])
#                 if 'G_I_L1_FLAIR' in losses:
#                     L1_FLAIR_test.append(losses['G_I_L1_FLAIR'])
#                 if 'G_I_L2_T1w' in losses:
#                     L2_T1w_test.append(losses['G_I_L2_T1w'])
#                 if 'G_I_L2_T2w' in losses:
#                     L2_T2w_test.append(losses['G_I_L2_T2w'])
#                 if 'G_I_L2_FLAIR' in losses:
#                     L2_FLAIR_test.append(losses['G_I_L2_FLAIR'])

#                 # Record image quality metrics: SSIM and PSNR per modality
#                 for key, target_list in zip(
#                     ['SSIM_T1w', 'SSIM_T2w', 'SSIM_FLAIR'],
#                     [SSIM_T1w_test, SSIM_T2w_test, SSIM_FLAIR_test]
#                 ):
#                     if key in losses:
#                         target_list.append(losses[key])
#                 for key, target_list in zip(
#                     ['PSNR_T1w', 'PSNR_T2w', 'PSNR_FLAIR'],
#                     [PSNR_T1w_test, PSNR_T2w_test, PSNR_FLAIR_test]
#                 ):
#                     if key in losses:
#                         target_list.append(losses[key])
                        
#                 # Additional metrics
#                 if 'vgg' in losses:
#                     vgg_test.append(losses['vgg'])
#                 if 'PDT1_relation' in losses:
#                     PDT1_relation_test.append(losses['PDT1_relation'])
#                 if 'PD_wm' in losses:
#                     PD_wm_test.append(losses['PD_wm'])
#                 if 'PD_variance' in losses:
#                     PD_variance_test.append(losses['PD_variance'])
#                 if 'PD_constraint_head' in losses:
#                     PD_constraint_head = losses['PD_constraint_head']
#                 if 'tv_reg' in losses:
#                     tv_reg = losses['tv_reg']

#                 rescaling_factor_T1w_test.append(
#                     np.mean(model.rescaling_factor_T1w.detach().cpu().numpy(), axis=0)
#                 )
#                 rescaling_factor_T2w_test.append(
#                     np.mean(model.rescaling_factor_T2w.detach().cpu().numpy(), axis=0)
#                 )
#                 rescaling_factor_FLAIR_test.append(
#                     np.mean(model.rescaling_factor_FLAIR.detach().cpu().numpy(), axis=0)
#                 )

#                 # Concatenate the outputs for qstar maps
#                 if all_PD is None:
#                     all_PD = model.Q1.detach().cpu().numpy()
#                     all_T1 = model.Q2.detach().cpu().numpy()
#                     all_T2 = model.Q3.detach().cpu().numpy()
#                 else:
#                     all_PD = np.concatenate((all_PD, model.Q1.detach().cpu().numpy()), axis=0)
#                     all_T1 = np.concatenate((all_T1, model.Q2.detach().cpu().numpy()), axis=0)
#                     all_T2 = np.concatenate((all_T2, model.Q3.detach().cpu().numpy()), axis=0)

#         # Reconstruct full output images from patches.
#         full_PD = aggregator_Q1.get_output_tensor().numpy()
#         full_T1 = aggregator_Q2.get_output_tensor().numpy() * 1000  # scale T1 to milliseconds
#         full_T2 = aggregator_Q3.get_output_tensor().numpy() * 1000  # scale T2 to milliseconds
#         full_T1w_fake = aggregator_T1w_fake.get_output_tensor().numpy()
#         full_T2w_fake = aggregator_T2w_fake.get_output_tensor().numpy()
#         full_FLAIR_fake = aggregator_FLAIR_fake.get_output_tensor().numpy()

#         if opt.save_val_nifti:
#             base_save_dir = os.path.join(test_results_dir, 'niftis')
#             # Create a subfolder for the current subject using its subject_id
#             subject_id = subject['subject_id']
#             subject_dir = os.path.join(base_save_dir, subject_id)
#             os.makedirs(subject_dir, exist_ok=True)
            
#             affine = subject['T1w'].affine  # using the T1w image’s affine as a reference

#             # Save generated qmaps as NIfTI files in the subject's folder.
#             save_path_PD = os.path.join(subject_dir, f'{subject_id}_qPD.nii.gz')
#             save_path_T1 = os.path.join(subject_dir, f'{subject_id}_qT1.nii.gz')
#             save_path_T2 = os.path.join(subject_dir, f'{subject_id}_qT2.nii.gz')
#             nib.save(nib.Nifti1Image(np.squeeze(full_PD), affine), save_path_PD)
#             nib.save(nib.Nifti1Image(np.squeeze(full_T1), affine), save_path_T1)
#             nib.save(nib.Nifti1Image(np.squeeze(full_T2), affine), save_path_T2)

#             # Save synthesized weighted images (output signal model) as NIfTI files.
#             save_path_T1w_fake = os.path.join(subject_dir, f'{subject_id}_fake_T1w.nii.gz')
#             save_path_T2w_fake = os.path.join(subject_dir, f'{subject_id}_fake_T2w.nii.gz')
#             save_path_FLAIR_fake = os.path.join(subject_dir, f'{subject_id}_fake_FLAIR.nii.gz')
#             nib.save(nib.Nifti1Image(np.squeeze(full_T1w_fake), affine), save_path_T1w_fake)
#             nib.save(nib.Nifti1Image(np.squeeze(full_T2w_fake), affine), save_path_T2w_fake)
#             nib.save(nib.Nifti1Image(np.squeeze(full_FLAIR_fake), affine), save_path_FLAIR_fake)

#             # # Save real images in the same subject-specific folder for easy comparison.
#             T1w_real = subject['T1w'].data.numpy() * subject['brain_mask'].data.numpy()
#             T2w_real = subject['T2w'].data.numpy() * subject['brain_mask'].data.numpy()
#             FLAIR_real = subject['FLAIR'].data.numpy() * subject['brain_mask'].data.numpy()
#             save_path_T1w_real = os.path.join(subject_dir, f'{subject_id}_real_T1w.nii.gz')
#             save_path_T2w_real = os.path.join(subject_dir, f'{subject_id}_real_T2w.nii.gz')
#             save_path_FLAIR_real = os.path.join(subject_dir, f'{subject_id}_real_FLAIR.nii.gz')
#             nib.save(nib.Nifti1Image(np.squeeze(T1w_real), affine), save_path_T1w_real)
#             nib.save(nib.Nifti1Image(np.squeeze(T2w_real), affine), save_path_T2w_real)
#             nib.save(nib.Nifti1Image(np.squeeze(FLAIR_real), affine), save_path_FLAIR_real)

#             if opt.eval_qstar_tissue_values:
#                 # Save tissue masks as NIfTI files
#                 wm_mask = subject['wm_mask'].data.numpy()
#                 gm_mask = subject['gm_mask'].data.numpy()
#                 csf_mask = subject['csf_mask'].data.numpy()
#                 save_path_wm_mask = os.path.join(subject_dir, f'{subject_id}_wm_mask.nii.gz')
#                 save_path_gm_mask = os.path.join(subject_dir, f'{subject_id}_gm_mask.nii.gz')
#                 save_path_csf_mask = os.path.join(subject_dir, f'{subject_id}_csf_mask.nii.gz')
#                 nib.save(nib.Nifti1Image(np.squeeze(wm_mask), affine), save_path_wm_mask)
#                 nib.save(nib.Nifti1Image(np.squeeze(gm_mask), affine), save_path_gm_mask)
#                 nib.save(nib.Nifti1Image(np.squeeze(csf_mask), affine), save_path_csf_mask)
            
#             mae_T1w_map = np.abs(full_T1w_fake - T1w_real)
#             mse_T1w_map = (full_T1w_fake - T1w_real) ** 2
#             mae_T2w_map = np.abs(full_T2w_fake - T2w_real)
#             mse_T2w_map = (full_T2w_fake - T2w_real) ** 2
#             mae_FLAIR_map = np.abs(full_FLAIR_fake - FLAIR_real)
#             mse_FLAIR_map = (full_FLAIR_fake - FLAIR_real) ** 2
#             error_nifti_dir = os.path.join(subject_dir, 'error_maps')
#             os.makedirs(error_nifti_dir, exist_ok=True)
#             save_path_mae_T1w = os.path.join(error_nifti_dir, f'{subject_id}_mae_T1w.nii.gz')
#             save_path_mse_T1w = os.path.join(error_nifti_dir, f'{subject_id}_mse_T1w.nii.gz')
#             save_path_mae_T2w = os.path.join(error_nifti_dir, f'{subject_id}_mae_T2w.nii.gz')
#             save_path_mse_T2w = os.path.join(error_nifti_dir, f'{subject_id}_mse_T2w.nii.gz')
#             save_path_mae_FLAIR = os.path.join(error_nifti_dir, f'{subject_id}_mae_FLAIR.nii.gz')
#             save_path_mse_FLAIR = os.path.join(error_nifti_dir, f'{subject_id}_mse_FLAIR.nii.gz')
#             nib.save(nib.Nifti1Image(np.squeeze(mae_T1w_map), affine), save_path_mae_T1w)
#             nib.save(nib.Nifti1Image(np.squeeze(mse_T1w_map), affine), save_path_mse_T1w)
#             nib.save(nib.Nifti1Image(np.squeeze(mae_T2w_map), affine), save_path_mae_T2w)
#             nib.save(nib.Nifti1Image(np.squeeze(mse_T2w_map), affine), save_path_mse_T2w)
#             nib.save(nib.Nifti1Image(np.squeeze(mae_FLAIR_map), affine), save_path_mae_FLAIR)
#             nib.save(nib.Nifti1Image(np.squeeze(mse_FLAIR_map), affine), save_path_mse_FLAIR)
            

#         if opt.eval_qstar_tissue_values:
#         # Get tissue masks (if available) from the subject for qstar calculation.
#             full_mask_wm = subject['wm_mask'].data.numpy().astype(bool)
#             full_mask_gm = subject['gm_mask'].data.numpy().astype(bool)
#             full_mask_csf = subject['csf_mask'].data.numpy().astype(bool)
#             full_mask_tumor = subject['tumor_mask'].data.numpy().astype(int) if 'tumor_mask' in subject.keys() else None

#             if opt.erode_masks:
#                 full_mask_wm = np.stack([binary_erosion(slice, iterations=1) for slice in full_mask_wm], axis=0)
#                 full_mask_gm = np.stack([binary_erosion(slice, iterations=1) for slice in full_mask_gm], axis=0)
#                 full_mask_csf = np.stack([binary_erosion(slice, iterations=1) for slice in full_mask_csf], axis=0)
                
#             if opt.aggregated_qstar_histograms:
#                 T1_wm_vals = full_T1[full_mask_wm]  # full_T1 was computed as: aggregator_Q2.get_output_tensor().numpy() * 1000
#                 T1_gm_vals = full_T1[full_mask_gm]
#                 T2_wm_vals = full_T2[full_mask_wm]  # full_T2 was computed similarly for T2 maps
#                 T2_gm_vals = full_T2[full_mask_gm]
#                 T1_wm_vals = T1_wm_vals[T1_wm_vals > 0.001]
#                 T1_gm_vals = T1_gm_vals[T1_gm_vals > 0.001]
#                 T2_wm_vals = T2_wm_vals[T2_wm_vals > 0.001]
#                 T2_gm_vals = T2_gm_vals[T2_gm_vals > 0.001]
#                 # Append values to the accumulated lists
#                 aggregated_T1_wm.append(T1_wm_vals)
#                 aggregated_T1_gm.append(T1_gm_vals)
#                 aggregated_T2_wm.append(T2_wm_vals)
#                 aggregated_T2_gm.append(T2_gm_vals)

#             # Compute the qstar values for the full output.
#             qstar_values = calc_values_qstarmaps(full_PD, full_T1, full_T2,
#                                                 full_mask_wm, full_mask_gm, full_mask_csf,
#                                                 full_mask_tumor)

#             test_qstar_values.loc[len(test_qstar_values)] = [
#                 subject['subject_id'],
#                 subject.get('protocol_id', 'NA'),
#                 *qstar_values
#             ]

#         # Aggregate the patch‐based metrics (averaging over patches) for this subject.
#         test_metrics.loc[len(test_metrics)] = [
#             subject['subject_id'],
#             np.mean(SSIM_T1w_test),
#             np.mean(SSIM_T2w_test),
#             np.mean(SSIM_FLAIR_test),
#             np.mean(PSNR_T1w_test),
#             np.mean(PSNR_T2w_test),
#             np.mean(PSNR_FLAIR_test),
#             np.mean(L1_list_test),
#             np.mean(L2_list_test),
#             np.mean(L1_T1w_test),
#             np.mean(L1_T2w_test),
#             np.mean(L1_FLAIR_test),
#             np.mean(L2_T1w_test),
#             np.mean(L2_T2w_test),
#             np.mean(L2_FLAIR_test),
#             np.mean(vgg_test),
#             np.mean(PDT1_relation_test),
#             np.mean(PD_wm_test),
#             np.mean(PD_variance_test),
#             np.mean(PD_constraint_head),
#             np.mean(tv_reg),
#             np.mean(rescaling_factor_T1w_test),
#             np.mean(rescaling_factor_T2w_test),
#             np.mean(rescaling_factor_FLAIR_test)
#         ]

#         # Optionally, save example images using the Visualizer.
#         if opt.save_images_visualizer:
#             real_T1w_slice    = visualizer.get_center_slice(subject['T1w'].data.numpy() * subject['brain_mask'].data.numpy())
#             real_T2w_slice    = visualizer.get_center_slice(subject['T2w'].data.numpy() * subject['brain_mask'].data.numpy())
#             real_FLAIR_slice = visualizer.get_center_slice(subject['FLAIR'].data.numpy() * subject['brain_mask'].data.numpy())
#             fake_T1w_slice    = visualizer.get_center_slice(full_T1w_fake)
#             fake_T2w_slice    = visualizer.get_center_slice(full_T2w_fake)
#             fake_FLAIR_slice = visualizer.get_center_slice(full_FLAIR_fake)
#             qPD_slice         = visualizer.get_center_slice(full_PD)
#             qT1_slice         = visualizer.get_center_slice(full_T1)
#             qT2_slice         = visualizer.get_center_slice(full_T2)
#             example_images_weighted_dir = os.path.join(test_results_dir, 'example_images_weighted')
#             os.makedirs(example_images_weighted_dir, exist_ok=True)
#             example_images_qstar_dir = os.path.join(test_results_dir, 'example_images_qstar')
#             os.makedirs(example_images_qstar_dir, exist_ok=True)
#             visualizer.save_examples_images(real_T1w_slice, real_T2w_slice, real_FLAIR_slice,
#                                             fake_T1w_slice, fake_T2w_slice, fake_FLAIR_slice,
#                                             qPD_slice, qT1_slice, qT2_slice,
#                                             save_loc=test_results_dir, epoch=subject['subject_id'],
#                                             save_weighted_loc=example_images_weighted_dir, save_qmap_loc=example_images_qstar_dir,
#                                             save_format='png')
            
#             # plot histograms of fake vs real
#             hist_dir_fake_real = os.path.join(test_results_dir, 'fake_real_histograms')
#             os.makedirs(hist_dir_fake_real, exist_ok=True)
#             hist_dir_qstar = os.path.join(test_results_dir, 'qstar_histograms')
#             os.makedirs(hist_dir_qstar, exist_ok=True)
#             # visualizer.plot_real_fake_kde(subject['T1w'].data.numpy() * subject['brain_mask'].data.numpy(), subject['T2w'].data.numpy() * subject['brain_mask'].data.numpy(), subject['FLAIR'].data.numpy() * subject['brain_mask'].data.numpy(),
#             #                                     full_T1w_fake * subject['brain_mask'].data.numpy(), full_T2w_fake * subject['brain_mask'].data.numpy(), full_FLAIR_fake *  subject['brain_mask'].data.numpy(),
#             #                                     save_loc=hist_dir_fake_real, epoch=subject['subject_id'])
#             if opt.eval_qstar_tissue_values:
#                 visualizer.plot_qstar_kde(full_PD, full_T1, full_T2, hist_dir_qstar, subject['subject_id'], wm_mask=full_mask_wm, gm_mask=full_mask_gm, csf_mask=full_mask_csf)
#             else:
#                 visualizer.plot_qstar_kde(full_PD, full_T1, full_T2, hist_dir_qstar, subject['subject_id'])
        
            
#             # visualizer.save_center_slice_plots(
#             #                     real_T1w=subject['T1w'].data.numpy(),
#             #                     real_T2w=subject['T2w'].data.numpy(),
#             #                     real_FLAIR=subject['FLAIR'].data.numpy(),
#             #                     fake_T1w=full_T1w_fake,
#             #                     fake_T2w=full_T2w_fake,
#             #                     fake_FLAIR=full_FLAIR_fake,
#             #                     qPD=full_PD,
#             #                     qT1=full_T1,
#             #                     qT2=full_T2,
#             #                     save_loc=test_results_dir,
#             #                     epoch=subject['subject_id']
#             #                 )
            
#     # Save overall test metrics and qstar values to CSV files.
#     test_metrics.to_csv(os.path.join(test_results_dir, 'test_metrics.csv'), index=False)
#     if opt.eval_qstar_tissue_values:
#         test_qstar_values.to_csv(os.path.join(test_results_dir, 'test_qstar_values_subjects.csv'), index=False)

#         # Compute overall summary statistics for qstar values over all subjects.
#         # Exclude non-numeric columns like 'subject_id' and 'protocol_id'.
#         qstar_numeric = test_qstar_values.drop(columns=['subject_id', 'protocol_id'] + [col for col in test_qstar_values.columns if 'std' in col])
#         overall_means = qstar_numeric.mean()
#         overall_stds = qstar_numeric.std()

#         # Create a summary DataFrame: each row represents a qstar measure.
#         qstar_summary = pd.DataFrame({
#             'qstar_measure': overall_means.index,
#             'overall_mean': overall_means.values,
#             'overall_std': overall_stds.values
#         })

#         # Save the summary DataFrame to a separate CSV file.
#         summary_csv_path = os.path.join(test_results_dir, 'qstar_overall_summary.csv')
#         qstar_summary.to_csv(summary_csv_path, index=False)
#         print("Overall qstar summary saved at:", summary_csv_path)
        
#         if opt.aggregated_qstar_histograms:
#             # plot histograms of qstar values
#             visualizer.plot_all_subjects_kde(aggregated_T1_wm, aggregated_T1_gm, aggregated_T2_wm, aggregated_T2_gm, hist_dir_qstar, 'all_test_subjects')
            
        
#     metrics_numeric = test_metrics.drop(columns=['subject_id'])
#     overall_metrics_mean = metrics_numeric.mean()
        
#     # Create a summary DataFrame for metrics where each row represents one metric.
#     metrics_summary = pd.DataFrame({
#         'metric': overall_metrics_mean.index,
#         'overall_mean': overall_metrics_mean.values
#     })
#     # Save the metrics summary to a separate CSV file.
#     metrics_summary_csv_path = os.path.join(test_results_dir, 'metrics_overall_summary.csv')
#     metrics_summary.to_csv(metrics_summary_csv_path, index=False)
#     print("Overall metrics summary saved at:", metrics_summary_csv_path)
        
#     print("Testing complete. Results saved at:", test_results_dir)


# if __name__ == '__main__':
#     main()
    
    
import os
import torch
import torchio as tio
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from qmap.util.util import SliceVisualizer
from qmap.options.test_options import TestOptions
from qmap.models import create_model
from qmap.data.conventional_dataset import create_dataset
from qmap.util.util import calc_qstar_stats, save_nifti


def test():
    opt = TestOptions().parse()
    opt.phase = 'test'
    
    dataset = create_dataset(opt)
    
    test_results_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir)
    os.makedirs(test_results_dir, exist_ok=True)
    
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qmap', 'util')
    # Create figure directory
    if opt.save_images_visualizer:
        visualizer = SliceVisualizer(script_dir)
        vis_dir = os.path.join(test_results_dir, 'output_figures')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Setup Model
    model = create_model(opt)
    model.setup(opt)
    print("Loading model weights from epoch:", opt.which_epoch_load)
    model.load_networks(opt.which_epoch_load)
    model.eval()

    # Containers for CSVs
    all_metrics_list = []
    all_qstar_list = []

    patch_size = (opt.padcrop, opt.padcrop, 1)

    print(f"Testing on {len(dataset)} subjects...")

    for subject in tqdm(dataset):
        subj_id = subject['subject_id']
        
        # Grid Sampling
        grid_sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap=(0,0,0))
        loader = torch.utils.data.DataLoader(grid_sampler, batch_size=opt.batchSize)
        
        # Aggregators
        agg_PD = tio.inference.GridAggregator(grid_sampler)
        agg_T1 = tio.inference.GridAggregator(grid_sampler)
        agg_T2 = tio.inference.GridAggregator(grid_sampler)
        
        agg_SynT1 = tio.inference.GridAggregator(grid_sampler)
        agg_SynT2 = tio.inference.GridAggregator(grid_sampler)
        agg_SynFLAIR = tio.inference.GridAggregator(grid_sampler)
        
        subj_metrics = {}
        patch_count = 0

        # --- 1. Patch Inference Loop ---
        for patch in loader:
            model.set_input(patch, phase_input='test')
            metrics = model.evaluate()
            
            # Aggregate Metrics
            for k, v in metrics.items():
                subj_metrics[k] = subj_metrics.get(k, 0.0) + v
            patch_count += 1
            
            # Collect Outputs
            locs = patch[tio.LOCATION]
            agg_PD.add_batch(model.Q1.detach().cpu().unsqueeze(-1), locs)
            agg_T1.add_batch(model.Q2.detach().cpu().unsqueeze(-1), locs)
            agg_T2.add_batch(model.Q3.detach().cpu().unsqueeze(-1), locs)
            
            agg_SynT1.add_batch(model.fake_T1w.detach().cpu().unsqueeze(-1), locs)
            agg_SynT2.add_batch(model.fake_T2w.detach().cpu().unsqueeze(-1), locs)
            agg_SynFLAIR.add_batch(model.fake_FLAIR.detach().cpu().unsqueeze(-1), locs)
            
        # --- 2. Process Metrics ---
        subj_metrics = {k: v / patch_count for k, v in subj_metrics.items()}
        subj_metrics['subject_id'] = subj_id
        all_metrics_list.append(subj_metrics)

        # --- 3. Process Full Volumes & Q-Star Stats ---
        full_PD = agg_PD.get_output_tensor().numpy().squeeze()
        full_T1 = agg_T1.get_output_tensor().numpy().squeeze() * 1000 # to ms
        full_T2 = agg_T2.get_output_tensor().numpy().squeeze() * 1000 # to ms
        
        # Get Synthetic Volumes
        syn_t1 = agg_SynT1.get_output_tensor().numpy().squeeze()
        syn_t2 = agg_SynT2.get_output_tensor().numpy().squeeze()
        syn_flair = agg_SynFLAIR.get_output_tensor().numpy().squeeze()

        # Get Real Volumes
        brain_mask = subject['brain_mask'][tio.DATA].numpy().squeeze()
        real_t1 = subject['T1w'][tio.DATA].numpy().squeeze() * brain_mask
        real_t2 = subject['T2w'][tio.DATA].numpy().squeeze() * brain_mask
        real_flair = subject['FLAIR'][tio.DATA].numpy().squeeze() * brain_mask
        
        if opt.eval_qstar_tissue_values:
            # Load masks (ensure bool for indexing in util function)
            m_wm = subject['wm_mask'][tio.DATA].numpy().squeeze().astype(bool)
            m_gm = subject['gm_mask'][tio.DATA].numpy().squeeze().astype(bool)
            m_csf = subject['csf_mask'][tio.DATA].numpy().squeeze().astype(bool)

            q_stats = calc_qstar_stats(full_PD, full_T1, full_T2, m_wm, m_gm, m_csf)
            q_stats['subject_id'] = subj_id
            all_qstar_list.append(q_stats)

        # --- 4. Save NIfTIs ---
        if opt.save_val_nifti:
            base_save_dir = os.path.join(test_results_dir, 'niftis')
            subject_dir = os.path.join(base_save_dir, subj_id)
            os.makedirs(subject_dir, exist_ok=True)
            affine = subject['T1w'].affine
            
            # Save Q-Maps
            save_nifti(full_PD, affine, os.path.join(subject_dir, f"{subj_id}_qPD.nii.gz"))
            save_nifti(full_T1, affine, os.path.join(subject_dir, f"{subj_id}_qT1.nii.gz"))
            save_nifti(full_T2, affine, os.path.join(subject_dir, f"{subj_id}_qT2.nii.gz"))
            
            # Save Synthetic
            save_nifti(agg_SynT1.get_output_tensor(), affine, os.path.join(subject_dir, f"{subj_id}_fake_T1w.nii.gz"))
            save_nifti(agg_SynT2.get_output_tensor(), affine, os.path.join(subject_dir, f"{subj_id}_fake_T2w.nii.gz"))
            save_nifti(agg_SynFLAIR.get_output_tensor(), affine, os.path.join(subject_dir, f"{subj_id}_fake_FLAIR.nii.gz"))
            
            # Save Real (Masked)
            brain_mask = subject['brain_mask'][tio.DATA]
            save_nifti(real_t1, affine, os.path.join(subject_dir, f"{subj_id}_real_T1w.nii.gz"))
            save_nifti(real_t2, affine, os.path.join(subject_dir, f"{subj_id}_real_T2w.nii.gz"))
            save_nifti(real_flair, affine, os.path.join(subject_dir, f"{subj_id}_real_FLAIR.nii.gz"))
            
            # Save Masks
            if opt.eval_qstar_tissue_values:
                 if 'wm_mask' in subject: save_nifti(m_wm, affine, os.path.join(subject_dir, f"{subj_id}_wm_mask.nii.gz"))
                 if 'gm_mask' in subject: save_nifti(m_gm, affine, os.path.join(subject_dir, f"{subj_id}_gm_mask.nii.gz"))
                 if 'csf_mask' in subject: save_nifti(m_csf, affine, os.path.join(subject_dir, f"{subj_id}_csf_mask.nii.gz"))
                 
        # --- GENERATE VISUALIZATION FIGURE ---
        if opt.save_images_visualizer:
            save_path = os.path.join(vis_dir, f"{subj_id}_slice_plot.png")
            visualizer.save_subject_figure(
                save_path,
                real_imgs=[real_t1, real_t2, real_flair],
                fake_imgs=[syn_t1, syn_t2, syn_flair],
                q_maps=[full_PD, full_T1, full_T2],
                subject_id=subj_id
            )

    # --- 5. Generate and Save CSVs ---
    
    # CSV 1: Subject-wise Metrics
    df_metrics = pd.DataFrame(all_metrics_list)
    cols = ['subject_id'] + [c for c in df_metrics.columns if c != 'subject_id']
    df_metrics = df_metrics[cols]
    df_metrics.to_csv(os.path.join(test_results_dir, "test_metrics.csv"), index=False)
    
    # CSV 2: Overall Mean Metrics
    metrics_summary = df_metrics.drop(columns=['subject_id']).mean().reset_index()
    metrics_summary.columns = ['metric', 'overall_mean']
    metrics_summary.to_csv(os.path.join(test_results_dir, "metrics_overall_summary.csv"), index=False)

    if opt.eval_qstar_tissue_values and all_qstar_list:
        # CSV 3: Subject-wise Q-Star Values
        df_qstar = pd.DataFrame(all_qstar_list)
        cols_q = ['subject_id'] + [c for c in df_qstar.columns if c != 'subject_id']
        df_qstar = df_qstar[cols_q]
        df_qstar.to_csv(os.path.join(test_results_dir, "test_qstar_values_subjects.csv"), index=False)
        
        # CSV 4: Overall Q-Star Summary (Mean + Std of means)
        qstar_numeric = df_qstar.drop(columns=['subject_id'])
        qstar_means = qstar_numeric.mean()
        qstar_stds = qstar_numeric.std()
        
        qstar_summary = pd.DataFrame({
            'qstar_measure': qstar_means.index,
            'overall_mean': qstar_means.values,
            'overall_std': qstar_stds.values
        })
        qstar_summary.to_csv(os.path.join(test_results_dir, "qstar_overall_summary.csv"), index=False)

    print(f"Testing Complete. Results saved to {test_results_dir}")

if __name__ == '__main__':
    test()