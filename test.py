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
    
    # Setup Model and loading of model weights
    model = create_model(opt)
    model.setup(opt)
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