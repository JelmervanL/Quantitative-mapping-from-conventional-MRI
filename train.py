# import math
# import random
# import time
# import os
# import gc
# import numpy as np
# import pandas as pd
# import torch.utils
# from tqdm import tqdm
# from scipy.ndimage import binary_erosion
# from Qstar.options.train_options import TrainOptions
# from Qstar.options.test_options import TestOptions
# from Qstar.models import create_model
# from Qstar.util.visualizer import Visualizer
# from Qstar.data.conventional_dataset import create_dataset, create_dataset_multiprot_cond
# import matplotlib
# import torch
# import torchio as tio
# matplotlib.use('Agg')  # non-interactive
# import nibabel as nib
# import wandb  

# def train_evaluate(opt, save_and_print=True):
#     # Set random seed for reproducibility
#     def set_seed(seed):
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)

#     set_seed(opt.random_seed)
    
#     # Initialize wandb
#     if opt.use_wandb:
#         wandb.init(
#             project=f"Qstar",
#             name=opt.name,
#             config={
#                 "architecture": opt.which_model_netG,
#                 "dataset": opt.dataroot,
#                 "learning_rate": opt.lr,
#                 "epochs": opt.n_epochs + opt.niter_decay,
#                 "batch_size": opt.batchSize,
#                 "input_modalities": opt.input_scan_types,
#                 "rescaling_method": opt.rescaling_method,
#                 "rescaling_factor_bounds": opt.rescaling_factor_bounds,
#                 "loss_L1": opt.loss_content_I_l1,
#                 "loss_L2": opt.loss_content_I_l2,
#                 "loss_vgg": opt.loss_vgg,
#                 "loss_pearson": opt.loss_content_pearson,
#                 "loss_PD_constraint": opt.loss_PD_constraint,
#                 "loss_tv_reg": opt.loss_tv_reg,
#                 "single_subject": opt.single_subject,
#                 "n_warmup_epochs": opt.n_warmup_epochs,
#                 'max_queuelength': opt.max_queuelength,
#                 "patches_per_volume": opt.patches_per_volume,
#                 'continue_train': opt.continue_train,
#             }
#         )
#         # Log all hyperparameters from the opt
#         wandb.config.update({k: v for k, v in vars(opt).items() if isinstance(v, (int, float, str, bool, list))})

#     # Will we do any logging (local or wandb)?
#     compute_logging_metrics = (opt.save_local or opt.use_wandb)
    
#     web_dir = os.path.join(opt.checkpoints_dir, opt.name)
#     example_images_save_train = os.path.join(web_dir, 'example_images', 'train')
#     os.makedirs(example_images_save_train, exist_ok=True)
    

#     dataset = create_dataset(opt)
#     dataset_size = len(dataset)
#     print('#training set size #of subjects = %d' % dataset_size)
        
#     # For validation
#     if opt.do_val:
#         opt2 = TestOptions().parse()
#         opt2.phase = 'val'
#         opt2.batchSize = opt.batchSize          
#         opt2.checkpoints_dir = opt.checkpoints_dir

#         if opt.multi_protocol_conditioning:
#             dataset_val = create_dataset_multiprot_cond(opt2)
#         else:
#             dataset_val = create_dataset(opt2)
#         dataset_size_val = len(dataset_val)
#         print('#validation set size #of subjects = %d' % dataset_size_val)

#         example_images_save_val = os.path.join(opt.checkpoints_dir, opt.name, 'example_images', 'val')
#         os.makedirs(example_images_save_val, exist_ok=True)
#         if opt.save_local:
#             val_metrics_results_dir = os.path.join(opt.checkpoints_dir, opt.name, 'val_metrics')
#             os.makedirs(val_metrics_results_dir, exist_ok=True)
#             csv_name_val_metrics = os.path.join(val_metrics_results_dir, 'val_metrics.csv')
#             qstar_values_name_val = os.path.join(val_metrics_results_dir, 'qstar_values.csv')
#         else:
#             csv_name_val_metrics = None
#             qstar_values_name_val = None
#     else:
#         print("Validation disabled (--do_val not set).")
#         dataset_val = None
        
#     # save options
#     web_dir = os.path.join(opt.checkpoints_dir, opt.name)
#     opt_file_name_train = os.path.join(web_dir, 'opt_train.txt')
#     with open(opt_file_name_train, 'wt') as opt_file:
#         for key, value in vars(opt).items():
#             opt_file.write(f'{key}: {value}\n')
#     if opt.do_val:
#         opt_file_name_val = os.path.join(web_dir, 'opt_val.txt')
#         with open(opt_file_name_val, 'wt') as opt_file:
#             for key, value in vars(opt2).items():
#                 opt_file.write(f'{key}: {value}\n')
            
#     # --- compute_logging_metrics check ---
#     # Create DataFrames only if we need them for wandb or local logging
#     if compute_logging_metrics:
#         train_results = pd.DataFrame(columns=[
#             'Epoch',
#             'SSIM_T1w','SSIM_T2w','SSIM_FLAIR',
#             'PSNR_T1w','PSNR_T2w','PSNR_FLAIR',
#             'L1_overall','L2_overall','Pearson_overall',
#             'L1_T1w','L1_T2w','L1_FLAIR',
#             'L2_T1w','L2_T2w','L2_FLAIR','Pearson_T1w','Pearson_T2w','Pearson_FLAIR',
#             'vgg','PDT1_relation','PD_wm','PD_variance','PD_constraint_head','tv_reg', 'PD_prior',
#             'rescaling_factor_T1w','rescaling_factor_T2w','rescaling_factor_FLAIR'
#         ])
#         validation_results = pd.DataFrame(columns=[
#             'Epoch',
#             'SSIM_T1w','SSIM_T2w','SSIM_FLAIR',
#             'PSNR_T1w','PSNR_T2w','PSNR_FLAIR',
#             'L1_overall','L2_overall','Pearson_overall',
#             'L1_T1w','L1_T2w','L1_FLAIR',
#             'L2_T1w','L2_T2w','L2_FLAIR','Pearson_T1w','Pearson_T2w','Pearson_FLAIR',
#             'vgg','PDT1_relation','PD_wm','PD_variance','PD_constraint_head','tv_reg',
#             'rescaling_factor_T1w','rescaling_factor_T2w','rescaling_factor_FLAIR'
#         ])
#         qstar_columns = [
#             'PD_wm_mean','PD_gm_mean','PD_csf_mean',
#             'PD_necrosis_mean','PD_enhancing_tumor_mean','PD_invasion_mean',
#             'T1_wm_mean','T1_gm_mean','T1_csf_mean','T1_necrosis_mean','T1_enhancing_tumor_mean','T1_invasion_mean',
#             'T2_wm_mean','T2_gm_mean','T2_csf_mean','T2_necrosis_mean','T2_enhancing_tumor_mean','T2_invasion_mean'
#         ]
#         validation_qstar_values_all = pd.DataFrame(columns=['Epoch'] + qstar_columns)
#         train_qstar_values = pd.DataFrame(columns=[
#             'Epoch','PD_wm_mean','PD_gm_mean','PD_csf_mean',
#             'PD_necrosis_mean','PD_enhancing_tumor_mean','PD_invasion_mean',
#             'T1_wm_mean','T1_gm_mean','T1_csf_mean','T1_necrosis_mean','T1_enhancing_tumor_mean','T1_invasion_mean',
#             'T2_wm_mean','T2_gm_mean','T2_csf_mean','T2_necrosis_mean','T2_enhancing_tumor_mean','T2_invasion_mean'
#         ])
#         if opt.save_local:
#             train_metrics_results_dir = os.path.join(opt.checkpoints_dir, opt.name, 'train_metrics')
#             os.makedirs(train_metrics_results_dir, exist_ok=True)
#             csv_name_train_metrics = os.path.join(train_metrics_results_dir, 'train_metrics.csv')
#             qstar_values_name_train = os.path.join(train_metrics_results_dir, 'qstar_values.csv')
#         else:
#             csv_name_train_metrics = None
#             qstar_values_name_train = None
#     else:
#         train_results = None
#         validation_results = None
#         validation_qstar_values_all = None
#         train_qstar_values = None
#         qstar_columns = None
#         csv_name_train_metrics = None
#         qstar_values_name_train = None
        
#     # setup model
#     model = create_model(opt)
#     model.setup(opt)  # from base_model
#     visualizer = Visualizer(opt)

#     # Set torchio queue and dataloader for training
#     patch_size = (224, 224, 1)
#     max_queue_length = opt.max_queuelength
#     patches_per_volume = opt.patches_per_volume
#     sampler = tio.data.UniformSampler(patch_size)
#     queue_train = tio.Queue(
#         dataset, max_length=max_queue_length, samples_per_volume=patches_per_volume,
#         sampler=sampler, shuffle_subjects=True, shuffle_patches=True, num_workers=4, start_background=True)
#     loader_train = torch.utils.data.DataLoader(queue_train, batch_size=opt.batchSize, num_workers=0)
    
#     epoch = 0
#     model.set_epoch(epoch)  
#     # warm-up on literature WM values
#     if not opt.continue_train and opt.n_warmup_epochs > 0:
#         print(f'Warm up start: {opt.n_warmup_epochs} epoch(s) of supervised training on literature WM vales')
#         for warmup_epoch in tqdm(range(opt.n_warmup_epochs), desc="Warmup epochs"):
#             for i, data in enumerate(tqdm(loader_train)):
#                 model.set_input(data)
#                 model.warmup_optimize_parameters()
#         print('Warm up end')
        
#     print('Training start')
#     total_steps = 0
#     for epoch in range(opt.epoch_count_start, opt.n_epochs + opt.niter_decay + 1):
#         model.set_epoch(epoch)          
#         epoch_start_time = time.time()
        
#         # If we are not logging metrics, skip the entire creation of running_metrics
#         if compute_logging_metrics:
#             batch_count = 0
#             running_metrics = {
#                 'L1':0.0, 'L2':0.0,
#                 'L1_T1w':0.0, 'L1_T2w':0.0, 'L1_FLAIR':0.0,
#                 'L2_T1w':0.0, 'L2_T2w':0.0, 'L2_FLAIR':0.0,
#                 'Pearson_T1w':0.0, 'Pearson_T2w':0.0, 'Pearson_FLAIR':0.0,
#                 'Pearson_overall':0.0,  # Overall Pearson loss
#                 'SSIM_T1w':0.0, 'SSIM_T2w':0.0, 'SSIM_FLAIR':0.0,
#                 'PSNR_T1w':0.0,'PSNR_T2w':0.0,'PSNR_FLAIR':0.0,
#                 'vgg':0.0,'PDT1_relation':0.0,'PD_wm':0.0,'PD_variance':0.0,'PD_constraint_head':0.0, 'tv_reg':0.0, 'PD_prior':0.0,
#                 'rescaling_factor_T1w':0.0,'rescaling_factor_T2w':0.0,'rescaling_factor_FLAIR':0.0
#             }
#             # For qstar computations
#             all_PD, all_T1, all_T2 = None, None, None
#             all_mask_wm, all_mask_gm, all_mask_csf, all_mask_tumor = None, None, None, None
#         else:
#             running_metrics = None
            
        
#         for i, data in enumerate(tqdm(loader_train)):
#             # We always do forward/backward passes, regardless of logging
#             if compute_logging_metrics:
#                 batch_count += 1
#             if total_steps % opt.print_freq == 0:
#                 t_data = time.time() - total_steps  # Not crucial, can remove or keep
#             total_steps += opt.batchSize

#             # Forward and backward update
#             model.set_input(data)
#             model.optimize_parameters()
            
#             # --- If we are not logging any metrics, skip calling model.get_current_losses():
#             if compute_logging_metrics:
#                 losses = model.get_current_losses()

#                 # Update overall losses using running average
#                 if 'G_I_L1' in losses:
#                     running_metrics['L1'] += (losses['G_I_L1'] - running_metrics['L1']) / batch_count
#                 if 'G_I_L2' in losses:
#                     running_metrics['L2'] += (losses['G_I_L2'] - running_metrics['L2']) / batch_count
#                 if 'G_I_pearson' in losses:
#                     running_metrics['Pearson_overall'] += (losses['G_I_pearson'] - running_metrics['Pearson_overall']) / batch_count

#                 # Update per-scan losses
#                 if 'G_I_L1_T1w' in losses:
#                     running_metrics['L1_T1w'] += (losses['G_I_L1_T1w'] - running_metrics['L1_T1w']) / batch_count
#                 if 'G_I_L1_T2w' in losses:
#                     running_metrics['L1_T2w'] += (losses['G_I_L1_T2w'] - running_metrics['L1_T2w']) / batch_count
#                 if 'G_I_L1_FLAIR' in losses:
#                     running_metrics['L1_FLAIR'] += (losses['G_I_L1_FLAIR'] - running_metrics['L1_FLAIR']) / batch_count
#                 if 'G_I_L2_T1w' in losses:
#                     running_metrics['L2_T1w'] += (losses['G_I_L2_T1w'] - running_metrics['L2_T1w']) / batch_count
#                 if 'G_I_L2_T2w' in losses:
#                     running_metrics['L2_T2w'] += (losses['G_I_L2_T2w'] - running_metrics['L2_T2w']) / batch_count
#                 if 'G_I_L2_FLAIR' in losses:
#                     running_metrics['L2_FLAIR'] += (losses['G_I_L2_FLAIR'] - running_metrics['L2_FLAIR']) / batch_count
#                 if 'G_I_pearson_T1w' in losses:
#                     running_metrics['Pearson_T1w'] += (losses['G_I_pearson_T1w'] - running_metrics['Pearson_T1w']) / batch_count
#                 if 'G_I_pearson_T2w' in losses:
#                     running_metrics['Pearson_T2w'] += (losses['G_I_pearson_T2w'] - running_metrics['Pearson_T2w']) / batch_count
#                 if 'G_I_pearson_FLAIR' in losses:
#                     running_metrics['Pearson_FLAIR'] += (losses['G_I_pearson_FLAIR'] - running_metrics['Pearson_FLAIR']) / batch_count
                    

#                 # Update additional metrics (SSIM, PSNR, etc.)
#                 for key in ['SSIM_T1w','SSIM_T2w','SSIM_FLAIR',
#                             'PSNR_T1w','PSNR_T2w','PSNR_FLAIR',
#                             'vgg','PDT1_relation','PD_wm','PD_variance','PD_constraint_head', 'tv_reg', 'PD_prior']:
#                     if key in losses:
#                         running_metrics[key] += (losses[key] - running_metrics[key]) / batch_count

#                 # Update rescaling factors from the model
#                 for key in ['rescaling_factor_T1w','rescaling_factor_T2w','rescaling_factor_FLAIR']:
#                     value = np.mean(getattr(model, key).detach().cpu().numpy())
#                     running_metrics[key] += (value - running_metrics[key]) / batch_count

#                 # If evaluating qstar, accumulate outputs
#                 if opt.eval_qstar_tissue_values:
#                     q1_cpu = model.Q1.detach().cpu().numpy()
#                     q2_cpu = model.Q2.detach().cpu().numpy()
#                     q3_cpu = model.Q3.detach().cpu().numpy()
#                     all_PD = np.concatenate((all_PD, q1_cpu), axis=0) if all_PD is not None else q1_cpu
#                     all_T1 = np.concatenate((all_T1, q2_cpu), axis=0) if all_T1 is not None else q2_cpu
#                     all_T2 = np.concatenate((all_T2, q3_cpu), axis=0) if all_T2 is not None else q3_cpu

#                     mask_wm_cpu = model.mask_wm.detach().cpu().numpy()
#                     mask_gm_cpu = model.mask_gm.detach().cpu().numpy()
#                     mask_csf_cpu = model.mask_csf.detach().cpu().numpy()
#                     mask_tumor_cpu = model.mask_tumor.detach().cpu().numpy().astype(int)

#                     if opt.erode_masks:
#                         mask_wm_cpu = binary_erosion(mask_wm_cpu.astype(bool), iterations=1)
#                         mask_gm_cpu = binary_erosion(mask_gm_cpu.astype(bool), iterations=1)
#                         mask_csf_cpu = binary_erosion(mask_csf_cpu.astype(bool), iterations=1)

#                     all_mask_wm = np.concatenate((all_mask_wm, mask_wm_cpu), axis=0) if all_mask_wm is not None else mask_wm_cpu
#                     all_mask_gm = np.concatenate((all_mask_gm, mask_gm_cpu), axis=0) if all_mask_gm is not None else mask_gm_cpu
#                     all_mask_csf = np.concatenate((all_mask_csf, mask_csf_cpu), axis=0) if all_mask_csf is not None else mask_csf_cpu
#                     all_mask_tumor = np.concatenate((all_mask_tumor, mask_tumor_cpu), axis=0) if all_mask_tumor is not None else mask_tumor_cpu
                    
#         # Optionally save model checkpoints 
#         if epoch % opt.save_latest_freq == 0 and save_and_print:
#             print(f'saving the latest model (epoch {epoch}, total_steps {total_steps})')
#             model.save_networks('latest')
#         if epoch % opt.save_epoch_freq == 0 and save_and_print:
#             print(f'saving the model at the end of epoch {epoch}, iters {total_steps}')
#             model.save_networks(epoch)             # only the epoch-specific file
            
#         # --- At end of epoch, log metrics if needed ---
#         if compute_logging_metrics:
#             # Update train_results dataframe
#             train_results.loc[epoch] = [
#                 epoch,
#                 running_metrics['SSIM_T1w'],
#                 running_metrics['SSIM_T2w'],
#                 running_metrics['SSIM_FLAIR'],
#                 running_metrics['PSNR_T1w'],
#                 running_metrics['PSNR_T2w'],
#                 running_metrics['PSNR_FLAIR'],
#                 running_metrics['L1'],
#                 running_metrics['L2'],
#                 running_metrics['Pearson_overall'],
#                 running_metrics['L1_T1w'],
#                 running_metrics['L1_T2w'],
#                 running_metrics['L1_FLAIR'],
#                 running_metrics['L2_T1w'],
#                 running_metrics['L2_T2w'],
#                 running_metrics['L2_FLAIR'],
#                 running_metrics['Pearson_T1w'],
#                 running_metrics['Pearson_T2w'],
#                 running_metrics['Pearson_FLAIR'],
#                 running_metrics['vgg'],
#                 running_metrics['PDT1_relation'],
#                 running_metrics['PD_wm'],
#                 running_metrics['PD_variance'],
#                 running_metrics['PD_constraint_head'],
#                 running_metrics['tv_reg'],
#                 running_metrics['PD_prior'],
#                 running_metrics['rescaling_factor_T1w'],
#                 running_metrics['rescaling_factor_T2w'],
#                 running_metrics['rescaling_factor_FLAIR'],
#             ]
#             # Save locally if needed
#             if opt.save_local:
#                 train_results.to_csv(csv_name_train_metrics, index=False)
#             if epoch % opt.save_latest_freq == 0: # always save images
#                 visualizer.save_examples_images(
#                     model.real_T1w, model.real_T2w, model.real_FLAIR,
#                     model.fake_T1w, model.fake_T2w, model.fake_FLAIR,
#                     model.Q1, model.Q2, model.Q3,
#                     save_loc=example_images_save_train, epoch=epoch
#                 )
#             # wandb logging
#             if opt.use_wandb:
#                 wandb_log_dict = {
#                     "epoch": epoch,
#                     "L1_overall/train": running_metrics['L1'],
#                     "L2_overall/train": running_metrics['L2'],
#                     "Pearson_overall/train": running_metrics['Pearson_overall'],
#                     "L1_T1w/train": running_metrics['L1_T1w'],
#                     "L1_T2w/train": running_metrics['L1_T2w'],
#                     "L1_FLAIR/train": running_metrics['L1_FLAIR'],
#                     "L2_T1w/train": running_metrics['L2_T1w'],
#                     "L2_T2w/train": running_metrics['L2_T2w'],
#                     "L2_FLAIR/train": running_metrics['L2_FLAIR'],
#                     "Pearson_T1w/train": running_metrics['Pearson_T1w'],
#                     "Pearson_T2w/train": running_metrics['Pearson_T2w'],
#                     "Pearson_FLAIR/train": running_metrics['Pearson_FLAIR'],
#                     "vgg/train": running_metrics['vgg'],
#                     "PDT1_relation/train": running_metrics['PDT1_relation'],
#                     "PD_wm/train": running_metrics['PD_wm'],
#                     "PD_variance/train": running_metrics['PD_variance'],
#                     "PD_constraint_head/train": running_metrics['PD_constraint_head'],
#                     "tv_reg/train": running_metrics['tv_reg'],
#                     "PD_prior/train": running_metrics['PD_prior'],
#                     "rescaling_factor_T1w/train": running_metrics['rescaling_factor_T1w'],
#                     "rescaling_factor_T2w/train": running_metrics['rescaling_factor_T2w'],
#                     "rescaling_factor_FLAIR/train": running_metrics['rescaling_factor_FLAIR'],
#                     "SSIM_T1w/train": running_metrics['SSIM_T1w'],
#                     "SSIM_T2w/train": running_metrics['SSIM_T2w'],
#                     "SSIM_FLAIR/train": running_metrics['SSIM_FLAIR'],
#                     "PSNR_T1w/train": running_metrics['PSNR_T1w'],
#                     "PSNR_T2w/train": running_metrics['PSNR_T2w'],
#                     "PSNR_FLAIR/train": running_metrics['PSNR_FLAIR'],
#                 }
#                 wandb.log(wandb_log_dict, step=epoch)
                
#             # If we want qstar
#             if opt.eval_qstar_tissue_values:
#                 PD_wm_mean, PD_gm_mean, PD_csf_mean, \
#                 T1_wm_mean, T1_gm_mean, T1_csf_mean, \
#                 T2_wm_mean, T2_gm_mean, T2_csf_mean, \
#                 PD_necrosis_mean, PD_enhancing_tumor_mean, PD_invasion_mean, \
#                 T1_necrosis_mean, T1_enhancing_tumor_mean, T1_invasion_mean, \
#                 T2_necrosis_mean, T2_enhancing_tumor_mean, T2_invasion_mean = \
#                     calc_values_qstarmaps(all_PD, all_T1, all_T2, all_mask_wm, all_mask_gm, all_mask_csf, all_mask_tumor)

#                 train_qstar_values.loc[epoch] = [
#                     epoch, PD_wm_mean, PD_gm_mean, PD_csf_mean,
#                     PD_necrosis_mean, PD_enhancing_tumor_mean, PD_invasion_mean,
#                     T1_wm_mean, T1_gm_mean, T1_csf_mean, T1_necrosis_mean, T1_enhancing_tumor_mean, T1_invasion_mean,
#                     T2_wm_mean, T2_gm_mean, T2_csf_mean, T2_necrosis_mean, T2_enhancing_tumor_mean, T2_invasion_mean
#                 ]
#                 if opt.save_local:
#                     train_qstar_values.to_csv(qstar_values_name_train, index=False)
#                 if opt.use_wandb:
#                     wandb_log_dict = {
#                         "qstar/PD/wm/train":  PD_wm_mean,
#                         "qstar/PD/gm/train":  PD_gm_mean,
#                         "qstar/PD/csf/train": PD_csf_mean,
#                         "qstar/T1/wm/train":  T1_wm_mean,
#                         "qstar/T1/gm/train":  T1_gm_mean,
#                         "qstar/T1/csf/train": T1_csf_mean,
#                         "qstar/T2/wm/train":  T2_wm_mean,
#                         "qstar/T2/gm/train":  T2_gm_mean,
#                         "qstar/T2/csf/train": T2_csf_mean,
#                     }
#                     wandb.log(wandb_log_dict, step=epoch)
                    
#         # update learning rate
#         print('End of epoch %d / %d \t Time Taken: %d sec' %
#               (epoch, opt.n_epochs + opt.niter_decay, time.time() - epoch_start_time))
#         model.update_learning_rate()
        
#          # --- Validation ---
#         if opt.do_val:
#             print('Validation start')
#             if compute_logging_metrics:
#                 val_batch_count = 0
#                 running_val_metrics = {
#                     'L1':0.0,'L2':0.0,
#                     'Pearson_overall':0.0,
#                     'L1_T1w':0.0,'L1_T2w':0.0,'L1_FLAIR':0.0,
#                     'L2_T1w':0.0,'L2_T2w':0.0,'L2_FLAIR':0.0,
#                     'Pearson_T1w':0.0,'Pearson_T2w':0.0,'Pearson_FLAIR':0.0,
#                     'SSIM_T1w':0.0,'SSIM_T2w':0.0,'SSIM_FLAIR':0.0,
#                     'PSNR_T1w':0.0,'PSNR_T2w':0.0,'PSNR_FLAIR':0.0,
#                     'vgg':0.0,'PDT1_relation':0.0,'PD_wm':0.0,'PD_variance':0.0,'PD_constraint_head':0.0, 'tv_reg':0.0,
#                     'rescaling_factor_T1w':0.0,'rescaling_factor_T2w':0.0,'rescaling_factor_FLAIR':0.0
#                 }
#             # per subject qstar values    
#             if compute_logging_metrics:
#                 validation_qstar_values = pd.DataFrame(columns=[
#                     'subject_id', 'protocol_id',
#                     'PD_wm_mean', 'PD_gm_mean', 'PD_csf_mean',
#                     'PD_necrosis_mean', 'PD_enhancing_tumor_mean', 'PD_invasion_mean',
#                     'T1_wm_mean', 'T1_gm_mean', 'T1_csf_mean', 'T1_necrosis_mean', 'T1_enhancing_tumor_mean', 'T1_invasion_mean',
#                     'T2_wm_mean', 'T2_gm_mean', 'T2_csf_mean', 'T2_necrosis_mean', 'T2_enhancing_tumor_mean', 'T2_invasion_mean'
#                 ])
    
                
#             for subject in (dataset_val if dataset_val else []):
#                 grid_sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap=(0,0,0), padding_mode=None)
#                 aggregator_Q1 = tio.inference.GridAggregator(grid_sampler,"crop")
#                 aggregator_Q2 = tio.inference.GridAggregator(grid_sampler,"crop")
#                 aggregator_Q3 = tio.inference.GridAggregator(grid_sampler,"crop")
#                 loader_val = torch.utils.data.DataLoader(grid_sampler, batch_size=opt.batchSize)

#                 for patch in loader_val:
#                     model.set_input(patch, phase_input='val')
#                     model.test()

#                     if compute_logging_metrics:
#                         val_batch_count += 1
#                         losses = model.get_current_losses()

#                         if 'G_I_L1' in losses:
#                             running_val_metrics['L1'] += (losses['G_I_L1'] - running_val_metrics['L1']) / val_batch_count
#                         if 'G_I_L2' in losses:
#                             running_val_metrics['L2'] += (losses['G_I_L2'] - running_val_metrics['L2']) / val_batch_count
#                         if 'G_I_pearson' in losses:
#                             running_val_metrics['Pearson_overall'] += (losses['G_I_pearson'] - running_val_metrics['Pearson_overall']) / val_batch_count
#                         if 'G_I_L1_T1w' in losses:
#                             running_val_metrics['L1_T1w'] += (losses['G_I_L1_T1w'] - running_val_metrics['L1_T1w']) / val_batch_count
#                         if 'G_I_L1_T2w' in losses:
#                             running_val_metrics['L1_T2w'] += (losses['G_I_L1_T2w'] - running_val_metrics['L1_T2w']) / val_batch_count
#                         if 'G_I_L1_FLAIR' in losses:
#                             running_val_metrics['L1_FLAIR'] += (losses['G_I_L1_FLAIR'] - running_val_metrics['L1_FLAIR']) / val_batch_count
#                         if 'G_I_L2_T1w' in losses:
#                             running_val_metrics['L2_T1w'] += (losses['G_I_L2_T1w'] - running_val_metrics['L2_T1w']) / val_batch_count
#                         if 'G_I_L2_T2w' in losses:
#                             running_val_metrics['L2_T2w'] += (losses['G_I_L2_T2w'] - running_val_metrics['L2_T2w']) / val_batch_count
#                         if 'G_I_L2_FLAIR' in losses:
#                             running_val_metrics['L2_FLAIR'] += (losses['G_I_L2_FLAIR'] - running_val_metrics['L2_FLAIR']) / val_batch_count
#                         if 'G_I_pearson_T1w' in losses:
#                             running_val_metrics['Pearson_T1w'] += (losses['G_I_pearson_T1w'] - running_val_metrics['Pearson_T1w']) / val_batch_count
#                         if 'G_I_pearson_T2w' in losses:
#                             running_val_metrics['Pearson_T2w'] += (losses['G_I_pearson_T2w'] - running_val_metrics['Pearson_T2w']) / val_batch_count
#                         if 'G_I_pearson_FLAIR' in losses:
#                             running_val_metrics['Pearson_FLAIR'] += (losses['G_I_pearson_FLAIR'] - running_val_metrics['Pearson_FLAIR']) / val_batch_count

#                         for k in ['SSIM_T1w','SSIM_T2w','SSIM_FLAIR','PSNR_T1w','PSNR_T2w','PSNR_FLAIR',
#                                   'vgg','PDT1_relation','PD_wm','PD_variance','PD_constraint_head', 'tv_reg']:
#                             if k in losses:
#                                 running_val_metrics[k] += (losses[k] - running_val_metrics[k]) / val_batch_count

#                         for k in ['rescaling_factor_T1w','rescaling_factor_T2w','rescaling_factor_FLAIR']:
#                             val_mean = np.mean(getattr(model, k).detach().cpu().numpy())
#                             running_val_metrics[k] += (val_mean - running_val_metrics[k]) / val_batch_count

#                         locations = patch[tio.LOCATION]
#                         # aggregator for Q1/Q2/Q3
#                         aggregator_Q1.add_batch(model.Q1.view(model.Q1.shape[0],1,*patch_size).detach().cpu(), locations)
#                         aggregator_Q2.add_batch(model.Q2.view(model.Q2.shape[0],1,*patch_size).detach().cpu(), locations)
#                         aggregator_Q3.add_batch(model.Q3.view(model.Q3.shape[0],1,*patch_size).detach().cpu(), locations)
                        
#                 if compute_logging_metrics:
#                     full_PD = aggregator_Q1.get_output_tensor().numpy()
#                     full_T1 = aggregator_Q2.get_output_tensor().numpy()
#                     full_T2 = aggregator_Q3.get_output_tensor().numpy()

#                     if opt.save_local and epoch % opt.save_latest_freq == 0 and opt.save_val_nifti:
#                         save_dir_example_nifti = os.path.join(example_images_save_val, 'niftis', f'epoch_{epoch}')
#                         os.makedirs(save_dir_example_nifti, exist_ok=True)
#                         affine = subject['T1w'].affine
#                         nib.save(nib.Nifti1Image(np.squeeze(full_PD), affine),
#                                  os.path.join(save_dir_example_nifti, f'{subject["subject_id"]}_qPD.nii.gz'))
#                         nib.save(nib.Nifti1Image(np.squeeze(full_T1), affine),
#                                  os.path.join(save_dir_example_nifti, f'{subject["subject_id"]}_qT1.nii.gz'))
#                         nib.save(nib.Nifti1Image(np.squeeze(full_T2), affine),
#                                  os.path.join(save_dir_example_nifti, f'{subject["subject_id"]}_qT2.nii.gz'))
            
#                     if opt.eval_qstar_tissue_values:
#                         full_mask_wm = subject['wm_mask'].data.numpy().astype(bool)
#                         full_mask_gm = subject['gm_mask'].data.numpy().astype(bool)
#                         full_mask_csf = subject['csf_mask'].data.numpy().astype(bool)
#                         full_mask_tumor = subject['tumor_mask'].data.numpy().astype(int)
#                         if opt.erode_masks:
#                             full_mask_wm   = np.stack([binary_erosion(s, iterations=1) for s in full_mask_wm], axis=0)
#                             full_mask_gm   = np.stack([binary_erosion(s, iterations=1) for s in full_mask_gm], axis=0)
#                             full_mask_csf  = np.stack([binary_erosion(s, iterations=1) for s in full_mask_csf], axis=0)

#                         (PD_wm_mean, PD_gm_mean, PD_csf_mean,
#                          T1_wm_mean, T1_gm_mean, T1_csf_mean,
#                          T2_wm_mean, T2_gm_mean, T2_csf_mean,
#                          PD_necrosis_mean, PD_enhancing_tumor_mean, PD_invasion_mean,
#                          T1_necrosis_mean, T1_enhancing_tumor_mean, T1_invasion_mean,
#                          T2_necrosis_mean, T2_enhancing_tumor_mean, T2_invasion_mean) = calc_values_qstarmaps(
#                              full_PD, full_T1, full_T2, full_mask_wm, full_mask_gm, full_mask_csf, full_mask_tumor)   
                         
#                         validation_qstar_values.loc[len(validation_qstar_values)] = [
#                             subject['subject_id'], subject['protocol_id'],
#                             PD_wm_mean, PD_gm_mean, PD_csf_mean,
#                             PD_necrosis_mean, PD_enhancing_tumor_mean, PD_invasion_mean,
#                             T1_wm_mean, T1_gm_mean, T1_csf_mean, T1_necrosis_mean, T1_enhancing_tumor_mean, T1_invasion_mean,
#                             T2_wm_mean, T2_gm_mean, T2_csf_mean, T2_necrosis_mean, T2_enhancing_tumor_mean, T2_invasion_mean]
                
#                 if compute_logging_metrics:
#                     if opt.eval_qstar_tissue_values:
#                         if opt.save_local:
#                                 csv_name_val_qstar_epoch = os.path.join(val_metrics_results_dir, f'qstar_values_subjects_{epoch}.csv')
#                                 validation_qstar_values.to_csv(csv_name_val_qstar_epoch, index=False)
#                         mean_values = validation_qstar_values[qstar_columns].mean()
#                         validation_qstar_values_all.to_csv(qstar_values_name_val, index=False)
#                         if opt.use_wandb:
#                             qstar_val_dict = {
#                                 # PD
#                                 "qstar/PD/wm/val":  mean_values['PD_wm_mean'],
#                                 "qstar/PD/gm/val":  mean_values['PD_gm_mean'],
#                                 "qstar/PD/csf/val": mean_values['PD_csf_mean'],

#                                 # T1
#                                 "qstar/T1/wm/val":  mean_values['T1_wm_mean'],
#                                 "qstar/T1/gm/val":  mean_values['T1_gm_mean'],
#                                 "qstar/T1/csf/val": mean_values['T1_csf_mean'],

#                                 # T2
#                                 "qstar/T2/wm/val":  mean_values['T2_wm_mean'],
#                                 "qstar/T2/gm/val":  mean_values['T2_gm_mean'],
#                                 "qstar/T2/csf/val": mean_values['T2_csf_mean'],

#                                 # And any tumor metrics if desired
#                             }
#                             wandb.log(qstar_val_dict, step=epoch)
#                         validation_qstar_values_all.loc[epoch] = [epoch] + mean_values.tolist()
                    
#                     validation_results.loc[epoch] = [
#                         epoch,
#                         running_val_metrics['SSIM_T1w'],
#                         running_val_metrics['SSIM_T2w'],
#                         running_val_metrics['SSIM_FLAIR'],
#                         running_val_metrics['PSNR_T1w'],
#                         running_val_metrics['PSNR_T2w'],
#                         running_val_metrics['PSNR_FLAIR'],
#                         running_val_metrics['L1'],
#                         running_val_metrics['L2'],
#                         running_val_metrics['Pearson_overall'],
#                         running_val_metrics['L1_T1w'],
#                         running_val_metrics['L1_T2w'],
#                         running_val_metrics['L1_FLAIR'],
#                         running_val_metrics['L2_T1w'],
#                         running_val_metrics['L2_T2w'],
#                         running_val_metrics['L2_FLAIR'],
#                         running_val_metrics['Pearson_T1w'],
#                         running_val_metrics['Pearson_T2w'],
#                         running_val_metrics['Pearson_FLAIR'],
#                         running_val_metrics['vgg'],
#                         running_val_metrics['PDT1_relation'],
#                         running_val_metrics['PD_wm'],
#                         running_val_metrics['PD_variance'],
#                         running_val_metrics['PD_constraint_head'],
#                         running_val_metrics['tv_reg'],
#                         running_val_metrics['rescaling_factor_T1w'],
#                         running_val_metrics['rescaling_factor_T2w'],
#                         running_val_metrics['rescaling_factor_FLAIR'],
#                     ]
#                     if opt.save_local:
#                         validation_results.to_csv(csv_name_val_metrics, index=False)
#                         visualizer.plot_losses(train_results, validation_results, os.path.join(opt.checkpoints_dir, opt.name))
#                         visualizer.plot_qstar_values(validation_qstar_values_all, os.path.join(opt.checkpoints_dir, opt.name))
#                     if opt.use_wandb:
#                         val_log_dict = {
#                             "epoch": epoch,
#                             "L1_overall/val": running_val_metrics['L1'],
#                             "L2_overall/val": running_val_metrics['L2'],
#                             "Pearson_overall/val": running_val_metrics['Pearson_overall'],
#                             "L1_T1w/val": running_val_metrics['L1_T1w'],
#                             "L1_T2w/val": running_val_metrics['L1_T2w'],
#                             "L1_FLAIR/val": running_val_metrics['L1_FLAIR'],
#                             "L2_T1w/val": running_val_metrics['L2_T1w'],
#                             "L2_T2w/val": running_val_metrics['L2_T2w'],
#                             "L2_FLAIR/val": running_val_metrics['L2_FLAIR'],
#                             "Pearson_T1w/val": running_val_metrics['Pearson_T1w'],
#                             "Pearson_T2w/val": running_val_metrics['Pearson_T2w'],
#                             "Pearson_FLAIR/val": running_val_metrics['Pearson_FLAIR'],
#                             "vgg/val": running_val_metrics['vgg'],
#                             "PDT1_relation/val": running_val_metrics['PDT1_relation'],
#                             "PD_wm/val": running_val_metrics['PD_wm'],
#                             "PD_variance/val": running_val_metrics['PD_variance'],
#                             "PD_constraint_head/val": running_val_metrics['PD_constraint_head'],
#                             "tv_reg/val": running_val_metrics['tv_reg'],
#                             "rescaling_factor_T1w/val": running_val_metrics['rescaling_factor_T1w'],
#                             "rescaling_factor_T2w/val": running_val_metrics['rescaling_factor_T2w'],
#                             "rescaling_factor_FLAIR/val": running_val_metrics['rescaling_factor_FLAIR'],
#                             "SSIM_T1w/val": running_val_metrics['SSIM_T1w'],
#                             "SSIM_T2w/val": running_val_metrics['SSIM_T2w'],
#                             "SSIM_FLAIR/val": running_val_metrics['SSIM_FLAIR'],
#                             "PSNR_T1w/val": running_val_metrics['PSNR_T1w'],
#                             "PSNR_T2w/val": running_val_metrics['PSNR_T2w'],
#                             "PSNR_FLAIR/val": running_val_metrics['PSNR_FLAIR'],
#                         }
#                         wandb.log(val_log_dict, step=epoch)
#                 # save images
#                 if epoch % opt.save_latest_freq == 0:
#                     visualizer.save_examples_images(
#                         model.real_T1w, model.real_T2w, model.real_FLAIR,
#                         model.fake_T1w, model.fake_T2w, model.fake_FLAIR,
#                         model.Q1, model.Q2, model.Q3,
#                         save_loc=example_images_save_val, epoch=epoch
#                     ) 
#         else:
#             if opt.save_local:
#                 visualizer.plot_losses(train_results, validation_results, os.path.join(opt.checkpoints_dir, opt.name))
                
#      # End of training loop
#     # finish wandb
#     if opt.use_wandb:
#         wandb.finish()
#     print(f'Finished training and evaluation of {opt.name}')

# # For validation: function to compute qstar values (unchanged)
# def calc_values_qstarmaps(all_PD, all_T1, all_T2, all_mask_wm, all_mask_gm, all_mask_csf, all_mask_tumor=None):
#     PD = all_PD.flatten().astype(np.float64)
#     T1 = all_T1.flatten().astype(np.float64)
#     T2 = all_T2.flatten().astype(np.float64)
#     mask_wm = all_mask_wm.flatten()
#     mask_gm = all_mask_gm.flatten()
#     mask_csf = all_mask_csf.flatten()

#     PD_wm, PD_gm, PD_csf = PD[mask_wm], PD[mask_gm], PD[mask_csf]
#     T1_wm, T1_gm, T1_csf = T1[mask_wm], T1[mask_gm], T1[mask_csf]
#     T2_wm, T2_gm, T2_csf = T2[mask_wm], T2[mask_gm], T2[mask_csf]

#     PD_wm_mean, PD_gm_mean, PD_csf_mean = np.mean(PD_wm), np.mean(PD_gm), np.mean(PD_csf)
#     T1_wm_mean, T1_gm_mean, T1_csf_mean = np.mean(T1_wm)*1000, np.mean(T1_gm)*1000, np.mean(T1_csf)*1000
#     T2_wm_mean, T2_gm_mean, T2_csf_mean = np.mean(T2_wm)*1000, np.mean(T2_gm)*1000, np.mean(T2_csf)*1000

#     PD_necrosis_mean, PD_enhancing_tumor_mean, PD_invasion_mean = None, None, None
#     T1_necrosis_mean, T1_enhancing_tumor_mean, T1_invasion_mean = None, None, None
#     T2_necrosis_mean, T2_enhancing_tumor_mean, T2_invasion_mean = None, None, None
#     if all_mask_tumor is not None:
#         mask_necrosis = (all_mask_tumor == 1).flatten()
#         mask_enhancing_tumor = (all_mask_tumor == 2).flatten()
#         mask_invasion = (all_mask_tumor == 4).flatten()

#         PD_necrosis, PD_enhancing_tumor, PD_invasion = PD[mask_necrosis], PD[mask_enhancing_tumor], PD[mask_invasion]
#         T1_necrosis, T1_enhancing_tumor, T1_invasion = T1[mask_necrosis], T1[mask_enhancing_tumor], T1[mask_invasion]
#         T2_necrosis, T2_enhancing_tumor, T2_invasion = T2[mask_necrosis], T2[mask_enhancing_tumor], T2[mask_invasion]

#         PD_necrosis_mean = np.mean(PD_necrosis)
#         PD_enhancing_tumor_mean = np.mean(PD_enhancing_tumor)
#         PD_invasion_mean = np.mean(PD_invasion)
#         T1_necrosis_mean = np.mean(T1_necrosis)*1000
#         T1_enhancing_tumor_mean = np.mean(T1_enhancing_tumor)*1000
#         T1_invasion_mean = np.mean(T1_invasion)*1000
#         T2_necrosis_mean = np.mean(T2_necrosis)*1000
#         T2_enhancing_tumor_mean = np.mean(T2_enhancing_tumor)*1000
#         T2_invasion_mean = np.mean(T2_invasion)*1000

#     return (PD_wm_mean, PD_gm_mean, PD_csf_mean,
#             T1_wm_mean, T1_gm_mean, T1_csf_mean,
#             T2_wm_mean, T2_gm_mean, T2_csf_mean,
#             PD_necrosis_mean, PD_enhancing_tumor_mean, PD_invasion_mean,
#             T1_necrosis_mean, T1_enhancing_tumor_mean, T1_invasion_mean,
#             T2_necrosis_mean, T2_enhancing_tumor_mean, T2_invasion_mean)

# if __name__ == '__main__':
#     opt = TrainOptions().parse()
#     train_evaluate(opt)
    
    
import time
import os
import numpy as np
import torch
import torchio as tio
from tqdm import tqdm
import wandb
import random
from scipy.ndimage import binary_erosion

from qmap.options.train_options import TrainOptions
from qmap.options.test_options import TestOptions
from qmap.models import create_model
from qmap.data.conventional_dataset import create_dataset
from qmap.util.util import set_seed, calc_qstar_stats 


def train(opt):
    set_seed(opt.random_seed)
    
    # --- 1. Setup WandB ---
    if opt.use_wandb:
        wandb.init(project=opt.wandb_project, name=opt.name, config=vars(opt))

    # --- 2. Training Data Setup ---
    dataset = create_dataset(opt)
    print(f'Training size: {len(dataset)} subjects')

    patch_size = (opt.padcrop, opt.padcrop, 1) 
    sampler = tio.data.UniformSampler(patch_size)
    queue_train = tio.Queue(
        dataset, max_length=opt.max_queuelength, samples_per_volume=opt.patches_per_volume,
        sampler=sampler, shuffle_subjects=True, shuffle_patches=True, num_workers=4
    )
    loader_train = torch.utils.data.DataLoader(queue_train, batch_size=opt.batchSize)

    # --- 3. Validation Data Setup (Run Once) ---
    dataset_val = None
    if opt.do_val:
        print("Setting up validation dataset (once)...")
        # Parse test options once to get defaults, then override with training consistency
        opt_val = TestOptions().parse() 
        opt_val.phase = 'val'
        opt_val.batchSize = opt.batchSize
        opt_val.dataroot = opt.dataroot 
        
 
        dataset_val = create_dataset(opt_val)
        print(f'Validation size: {len(dataset_val)} subjects')
    
    # --- 4. Model Setup ---
    model = create_model(opt)
    model.setup(opt)
    
    # Warmup for more stable training. supervised training on literature WM T1 T2 and PD maps
    if not opt.continue_train and opt.n_warmup_epochs > 0:
        print("Starting Warmup...")
        model.set_epoch(0) # Ensure epoch is set for warmup logic
        for _ in range(opt.n_warmup_epochs):
            for data in tqdm(loader_train, desc="Warmup"):
                model.set_input(data)
                model.warmup_optimize_parameters()

    # --- 5. Training Loop ---
    total_steps = 0
    
    for epoch in range(opt.epoch_count_start, opt.n_epochs + opt.niter_decay + 1):
        epoch_start = time.time()
        model.set_epoch(epoch)
        
        # Metric Accumulator for this epoch
        train_metrics_acc = {} 
        batch_count = 0

        # --- TRAIN PHASE ---
        for data in tqdm(loader_train, desc=f"Epoch {epoch} Train"):
            total_steps += opt.batchSize
            model.set_input(data)
            
            # Optimization step returns dictionary of metrics
            metrics = model.optimize_parameters() 
            for k, v in metrics.items():
                if np.isnan(v):
                    print(f"NaN detected in metric {k} at epoch {epoch}")

            # Aggregate metrics
            for k, v in metrics.items():
                train_metrics_acc[k] = train_metrics_acc.get(k, 0.0) + v
            batch_count += 1

        # Log Average Train Metrics
        if opt.use_wandb:
            log_dict = {f"Train/{k}": v / batch_count for k, v in train_metrics_acc.items()}
            log_dict['epoch'] = epoch
            wandb.log(log_dict, step=epoch)

        # --- VALIDATION PHASE ---
        # Pass the pre-initialized dataset_val
        if opt.do_val and dataset_val is not None:
            run_validation(model, opt, dataset_val, epoch, patch_size)

        # Checkpointing
        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving checkpoint at epoch {epoch}')
            model.save_networks(epoch)
            
        if epoch % opt.save_latest_freq == 0:
            model.save_networks('latest')

        model.update_learning_rate()
        print(f"Epoch {epoch} done in {time.time() - epoch_start:.2f}s")

    if opt.use_wandb:
        wandb.finish()
        
def run_validation(model, opt, dataset_val, epoch, patch_size):
    """
    Handles validation loop with corrected aggregation for patch-level 
    vs subject-level metrics.
    """
    print("Running Validation...")
    
    # --- 1. Separate Accumulators ---
    patch_metrics_acc = {}   # For losses (MSE, L1) calculated per patch
    subject_metrics_acc = {} # For Q-Star stats calculated per volume
    
    total_patches = 0
    total_subjects = 0
    
    # Validation Loop (Subject-based)
    for subject in tqdm(dataset_val, desc="Validation"):
        grid_sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap=(0,0,0))
        loader = torch.utils.data.DataLoader(grid_sampler, batch_size=opt.batchSize)
        
        # Aggregators for full volume reconstruction
        agg_Q1 = tio.inference.GridAggregator(grid_sampler)
        agg_Q2 = tio.inference.GridAggregator(grid_sampler)
        agg_Q3 = tio.inference.GridAggregator(grid_sampler)

        # --- Inner Loop: Patch Level ---
        for patch in loader:
            model.set_input(patch, phase_input='val')
            metrics = model.evaluate()

            # Accumulate Patch Metrics
            for k, v in metrics.items():
                patch_metrics_acc[k] = patch_metrics_acc.get(k, 0.0) + v
            
            # Increment Patch Counter
            total_patches += 1
            
            # Store outputs for stitching
            locs = patch[tio.LOCATION]
            agg_Q1.add_batch(model.Q1.detach().cpu().unsqueeze(-1), locs)
            agg_Q2.add_batch(model.Q2.detach().cpu().unsqueeze(-1), locs)
            agg_Q3.add_batch(model.Q3.detach().cpu().unsqueeze(-1), locs)

        # --- Outer Loop Processing: Subject Level ---
        vol_PD = agg_Q1.get_output_tensor().numpy().squeeze()
        vol_T1 = agg_Q2.get_output_tensor().numpy().squeeze()
        vol_T2 = agg_Q3.get_output_tensor().numpy().squeeze()
        
        if opt.eval_qstar_tissue_values:
            m_wm = subject['wm_mask'][tio.DATA].numpy().squeeze()
            m_gm = subject['gm_mask'][tio.DATA].numpy().squeeze()
            m_csf = subject['csf_mask'][tio.DATA].numpy().squeeze()
            
            m_wm = m_wm.astype(bool)
            m_gm = m_gm.astype(bool)
            m_csf = m_csf.astype(bool)

            # Collect data for Q-Star analysis
            q_stats = calc_qstar_stats(vol_PD, vol_T1, vol_T2, m_wm, m_gm, m_csf)
            
            if opt.use_wandb:
                for k, v in q_stats.items():
                    # Accumulate Subject Metrics
                    # We add the prefix here to keep the key unique
                    key = f"QStar/{k}"
                    subject_metrics_acc[key] = subject_metrics_acc.get(key, 0.0) + v
        
        # Increment Subject Counter
        total_subjects += 1

    # --- Logging ---
    if opt.use_wandb:
        log_dict = {}
        
        # 1. Log Patch Metrics (divide by total_patches)
        for k, v in patch_metrics_acc.items():
            log_dict[f"Val/{k}"] = v / total_patches
            
        # 2. Log Subject Metrics (divide by total_subjects)
        for k, v in subject_metrics_acc.items():
            # Key already contains "QStar/..." from the loop above
            log_dict[k] = v / total_subjects
            
        log_dict['epoch'] = epoch
        wandb.log(log_dict, step=epoch)

if __name__ == '__main__':
    # 1. Instantiate the parser class
    opt_parser = TrainOptions()
    
    # 2. Parse the arguments
    opt = opt_parser.parse()
    
    # 3. Create the output directory and save the config file
    # This calls util.mkdirs(expr_dir) internally
    opt_parser.print_options(opt)
    
    # 4. Start training
    train(opt)