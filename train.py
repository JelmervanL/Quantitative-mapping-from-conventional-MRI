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