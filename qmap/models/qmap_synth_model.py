import torch
import random
from .base_model import BaseModel
from . import networks
from . import losses
import torchio as tio

class QMapSynthModel(BaseModel):
    def name(self):
        return 'QMapSynthModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        self.model_names = ['G']
        self.visual_names = ['fake_T1w', 'real_T1w', 'fake_T2w', 'real_T2w', 'fake_FLAIR', 'real_FLAIR', 'Q1', 'Q2', 'Q3']
        
        self._current_epoch = 0

        # Define Generator
        self.netG = networks.define_G(self.opt, opt.input_nc, opt.output_nc, 
                                      opt.which_model_netG, opt.encoder_norm, opt.decoder_norm, 
                                      opt.bottleneck_norm, opt.init_type, opt.init_gain, 
                                      self.gpu_ids, opt.input_scan_types)

        # Losses
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        
        if opt.loss_vgg > 0:
            self.perceptual = losses.PerceptualLoss()
            self.perceptual.initialize(self.criterionMSE)

        # Optimizers
        if self.isTrain:
            if opt.optimizer == 'adam':
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            elif opt.optimizer == 'sgd':
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=opt.beta1, weight_decay=opt.weight_decay)
            elif opt.optimizer == 'adamw':
                self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers = [self.optimizer_G]

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch   

    def set_input(self, input, phase_input='train'):
        # Move inputs to device and handle dimensions
        self.mask_brain = input['brain_mask'][tio.DATA].to(self.device).squeeze(-1).bool()
        self.otsu_mask = input['otsu_mask'][tio.DATA].to(self.device).squeeze(-1).bool()
        
        if 'gm_mask' in input and 'wm_mask' in input:
            self.mask_gm = input['gm_mask'][tio.DATA].to(self.device).squeeze(-1).bool()
            self.mask_wm = input['wm_mask'][tio.DATA].to(self.device).squeeze(-1).bool()
            self.mask_gm_wm = self.mask_gm | self.mask_wm
        else:
            # Fallback if masks aren't loaded (e.g. inference without eval_tissue flags)
            self.mask_gm_wm = self.mask_brain
        
        self.real_T1w = input['T1w'][tio.DATA].to(dtype=torch.float, device=self.device).squeeze(-1) * self.mask_brain
        self.real_T2w = input['T2w'][tio.DATA].to(dtype=torch.float, device=self.device).squeeze(-1) * self.mask_brain
        self.real_FLAIR = input['FLAIR'][tio.DATA].to(dtype=torch.float, device=self.device).squeeze(-1) * self.mask_brain

        # Handle input selection
        scan_map = {'T1w': self.real_T1w, 'T2w': self.real_T2w, 'FLAIR': self.real_FLAIR}
        
        if 'shared_attn_unet' in self.opt.which_model_netG:
            self.G_inputs = self._prepare_inputs_with_dropout(scan_map)
        
        # Load Sequence parameters
        self.T1w_TR = input['T1w_TR'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.T1w_TE = input['T1w_TE'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.T1w_TI = input['T1w_TI'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.T1w_FA = input['T1w_FA'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.T2w_TR = input['T2w_TR'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.T2w_TE = input['T2w_TE'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.T2w_FA = input['T2w_FA'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.FLAIR_TR = input['FLAIR_TR'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.FLAIR_TE = input['FLAIR_TE'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.FLAIR_TI = input['FLAIR_TI'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.FLAIR_FA = input['FLAIR_FA'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        
        # Rescaling factors from metadata
        self.T1w_rescaling_factor = input['T1w_rescaling_factor'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.T2w_rescaling_factor = input['T2w_rescaling_factor'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.FLAIR_rescaling_factor = input['FLAIR_rescaling_factor'].unsqueeze(-1).to(dtype=torch.float, device=self.device)

        self.T1w_seq_type = input['T1w_seq_type']
        self.T2w_seq_type = input['T2w_seq_type']
        self.FLAIR_seq_type = input['FLAIR_seq_type']

    def _prepare_inputs_with_dropout(self, scan_map):
        """
        Selects and prepares input tensors for the generator.
        
        DURING TRAINING:
          - Applies random dropout based on 'modality_dropout_rate'.
          - Enforces 'modality_dropout_max_missing' (never drop too many).
          - Enforces at least one modality is always kept.
          - Shuffles the list to ensure the Shared Encoder is permutation invariant.
          
        DURING INFERENCE/VALIDATION:
          - Returns all requested modalities in the fixed order defined by options.
        """
        input_types = self.opt.input_scan_types
        M = len(input_types)
        device = self.device
        
        # 1. Determine Keep Flags (Which modalities to use?)
        # Start by keeping everything (True)
        keep_flags = torch.ones(M, dtype=torch.bool, device=device)

        # Only apply dropout logic if training AND rate > 0 AND past burn-in
        if self.isTrain and self.opt.modality_dropout_rate > 0.0:
            if self._current_epoch >= self.opt.modality_dropout_burnin:
                
                # A. Propose random drops
                # rand < rate means we DROP (False)
                drop_mask = torch.rand(M, device=device) < self.opt.modality_dropout_rate
                keep_flags = ~drop_mask # Invert so True means KEEP

                # B. Enforce Max Missing Constraint
                # If we dropped too many, random select some to rescue
                num_dropped = (~keep_flags).sum()
                max_missing = self.opt.modality_dropout_max_missing
                
                if num_dropped > max_missing:
                    # Get indices of modalities that are currently False (dropped)
                    dropped_indices = torch.where(~keep_flags)[0]
                    # How many do we need to turn back to True?
                    num_to_rescue = num_dropped - max_missing
                    # Randomly pick indices to rescue
                    rescue_idx = dropped_indices[torch.randperm(len(dropped_indices))[:num_to_rescue]]
                    keep_flags[rescue_idx] = True

                # C. Enforce "At Least One" Constraint
                # If everything is dropped, randomly pick one to keep
                if (~keep_flags).all():
                    rescue_idx = torch.randint(0, M, (1,), device=device)
                    keep_flags[rescue_idx] = True

        # 2. Build the Input List
        inputs = []
        for idx, scan_name in enumerate(input_types):
            # Only append if the flag for this index is True
            if keep_flags[idx]:
                if scan_name not in scan_map:
                    raise KeyError(f"Scan '{scan_name}' is in input_scan_types but missing from batch dictionary.")
                inputs.append(scan_map[scan_name])

        # 3. Shuffle for Permutation Invariance (Training Only)
        # We shuffle the list so the network doesn't learn that "Index 0 is always T1w".
        if self.isTrain:
            random.shuffle(inputs)

        return inputs

    def forward(self):
        # 1. Network Inference
        if 'shared_attn_unet' in self.opt.which_model_netG:
            self.fake_B = self.netG(self.G_inputs)
        else:
            # Fallback for other architectures
            inputs = [self.real_T1w if s == 'T1w' else self.real_T2w if s == 'T2w' else self.real_FLAIR for s in self.opt.input_scan_types]
            self.fake_B = self.netG(*inputs)
        
        self.Q1 = (self.fake_B[:, 0, :, :]*1)[:, None, :, :] * self.mask_brain  # PD
        self.Q2 = (self.fake_B[:, 1, :, :]*5)[:, None, :, :] * self.mask_brain  # T1 (scaled)
        self.Q3 = (self.fake_B[:, 2, :, :]*3)[:, None, :, :] * self.mask_brain  # T2 (scaled)
        
        # For further processing:
        self.PD = (self.fake_B[:, 0, :, :]*1) * self.mask_brain.squeeze(1)
        self.T1 = (self.fake_B[:, 1, :, :]*5) * self.mask_brain.squeeze(1)
        self.T2 = (self.fake_B[:, 2, :, :]*3) * self.mask_brain.squeeze(1)

        # 3. Physics-Based Signal Synthesis
        # Note: _batch_signal_model handles the loop over batch items for different sequence types
        self.fake_T1w_raw = self._batch_signal_model(self.PD, self.T1, self.T2, self.T1w_TR, self.T1w_TE, self.T1w_TI, self.T1w_FA, self.T1w_seq_type)
        self.fake_T2w_raw = self._batch_signal_model(self.PD, self.T1, self.T2, self.T2w_TR, self.T2w_TE, None, self.T2w_FA, self.T2w_seq_type)
        self.fake_FLAIR_raw = self._batch_signal_model(self.PD, self.T1, self.T2, self.FLAIR_TR, self.FLAIR_TE, self.FLAIR_TI, self.FLAIR_FA, self.FLAIR_seq_type)
        
    # Physical signal model
    def _batch_signal_model(self, PD, T1, T2, TR, TE, TI, FA, seq_types):
        outs = []
        B = PD.size(0)
        for b in range(B):
            out = self.signal_model(
                PD[b], T1[b], T2[b],
                TR[b], TE[b], None if TI is None else TI[b], FA[b],
                seq_type=str(seq_types[b]),          
            )
            out = out.squeeze(0)
            outs.append(out)
        return torch.stack(outs, dim=0)              # B×1×H×W

    def signal_model(self, PD, T1, T2, TR, TE, TI, FA, seq_type='GRE'):
        has_ti = TI is not None and not torch.isnan(TI).any()
        if has_ti:
            TI = TI / 1000
            TI = TI.unsqueeze(-1)
        TR, TE = TR / 1000, TE / 1000
        TR = TR.unsqueeze(-1)
        TE = TE.unsqueeze(-1)
        FA = FA.unsqueeze(-1)
        FA = torch.deg2rad(FA)
        
        if seq_type == 'MPRAGE':
            out = torch.abs(PD * (1 - 2 * torch.exp(-TI / (T1 + 1e-6)) + torch.exp(-TR / (T1 + 1e-6))) * torch.sin(FA) * torch.exp(-TE / (T2 + 1e-6)) / (1 + torch.cos(FA) * torch.exp(-TR / (T1 + 1e-6))))
        elif seq_type == 'GRE' or seq_type == 'TFE':
            out = PD * torch.sin(FA) * (1 - torch.exp(-TR / (T1+1e-6))) * torch.exp(-TE / (T2+1e-6)) / (1 - torch.cos(FA) * torch.exp(-TR / (T1+1e-6)))
        elif seq_type == 'SE':
            out =  PD * torch.sin(FA) * (1 - torch.exp(-TR / (T1+1e-6))) * torch.exp(-TE / (T2+1e-6)) / (1 - torch.cos(FA) * torch.exp(-TR / (T1+1e-6)))
        elif seq_type == 'TSE':
            out = PD * torch.exp(-TE / (T2 + 1e-6)) * (1 - torch.exp(-TR / (T1 + 1e-6)))
        elif seq_type == 'IR' or seq_type == 'FLAIR':
            out = torch.abs(PD * torch.exp(-TE / (T2 + 1e-6)) * (1 - 2 * torch.exp(-TI / (T1 + 1e-6)) + torch.exp(-TR / (T1 + 1e-6))))
        else:
            raise ValueError('Seq type  not recognized')
        if out.dim() == 2:              
            out = out.unsqueeze(0)      
        return out[:, None, :, :]      

    def _apply_rescaling(self, fake, real, bounds_percent, exact_factor):
        """Calculates and applies rescaling factor. Returns rescaled fake and the factor."""
        method = self.opt.rescaling_method
        
        # Pre-calculate bounds (used only if method is bounded)
        bounds = (exact_factor - bounds_percent * exact_factor, exact_factor + bounds_percent * exact_factor)
        lower, upper = bounds[0].squeeze(1), bounds[1].squeeze(1)

        factor = None

        if method == 'mean':
            mask = self.mask_gm_wm.float() 
            mean_fake = (fake * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)
            mean_real = (real * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)
            factor = mean_real / (mean_fake + 1e-6)
            factor = torch.clamp(factor, min=lower, max=upper)

        elif method == 'closed_form_least_squares':
            mask = self.mask_brain
            num = (mask * fake * real).sum(dim=(1, 2, 3))
            den = (mask * fake * fake).sum(dim=(1, 2, 3)).clamp(min=1e-6)
            factor = (num / den).detach()

        elif method == 'closed_form_least_squares_bounded':
            mask = self.mask_brain
            num = (mask * fake * real).sum(dim=(1, 2, 3))
            den = (mask * fake * fake).sum(dim=(1, 2, 3)).clamp(min=1e-6)
            factor = num / den
            # Apply bounds
            factor = torch.clamp(factor, min=lower, max=upper)


        elif method == 'fixed_per_subject':
            factor = exact_factor.squeeze(1)
        
        else:
            # Fallback to fixed factor if method not recognized
            factor = exact_factor.squeeze(1)

        # Reshape for broadcasting (B, 1, 1, 1)
        if factor.ndim == 1:
            factor = factor.view(-1, 1, 1, 1)
            
        return fake * factor, factor

    def compute_losses_and_metrics(self, optimize=False):
        """Unified function for calculating losses and metrics."""
        
        # 1. Setup Data for Loop
        scan_data = [
            ('T1w', self.fake_T1w_raw, self.real_T1w, self.T1w_rescaling_factor),
            ('T2w', self.fake_T2w_raw, self.real_T2w, self.T2w_rescaling_factor),
            ('FLAIR', self.fake_FLAIR_raw, self.real_FLAIR, self.FLAIR_rescaling_factor)
        ]

        total_loss_G = 0.0
        metrics = {}
        
        # Lists for aggregating averaged losses
        l1_losses, l2_losses, pearson_losses, vgg_losses = [], [], [], []

        # 2. Iterate over modalities
        for name, fake_raw, real, exact_factor in scan_data:
            # Rescale
            fake_rescaled, factor = self._apply_rescaling(fake_raw, real, self.opt.rescaling_factor_bounds, exact_factor)
            
            # Save rescaled versions for visualization
            setattr(self, f'fake_{name}', fake_rescaled) # Update visual attribute
            setattr(self, f'rescaling_factor_{name}', factor) # Save factor for analysis

            # Basic Metrics
            l1 = self.criterionL1(fake_rescaled, real)
            l2 = self.criterionMSE(fake_rescaled, real)
            pearson = losses.compute_pearson_loss(fake_raw, real) # Pearson is scale invariant usually, but raw/rescaled is fine

            # Store individual metrics
            metrics[f'L1_{name}'] = l1.item()
            metrics[f'L2_{name}'] = l2.item()
            metrics[f'Pearson_{name}'] = pearson.item()
            metrics[f'Factor_{name}'] = factor.mean().item()

            # Accumulate for optimization ONLY if this modality is a training target
            if name in self.opt.input_scan_types:
                l1_losses.append(l1)
                l2_losses.append(l2)
                pearson_losses.append(pearson)
                
                if self.opt.loss_vgg > 0:
                    vgg = self.perceptual.get_loss(fake_rescaled, real)
                    vgg_losses.append(vgg)
                    metrics[f'VGG_{name}'] = vgg.item()

            # Evaluation Metrics (SSIM/PSNR) - Calc if validation or requested
            if not optimize or self.opt.calc_additional_metrics:
                with torch.no_grad():
                    metrics[f'SSIM_{name}'] = losses.SSIM_crop(real, fake_rescaled, self.mask_brain).item()

        # 3. Aggregate Global Content Losses
        loss_l1_agg = torch.stack(l1_losses).mean() if l1_losses else torch.tensor(0.0, device=self.device)
        loss_l2_agg = torch.stack(l2_losses).mean() if l2_losses else torch.tensor(0.0, device=self.device)
        loss_pearson_agg = torch.stack(pearson_losses).mean() if pearson_losses else torch.tensor(0.0, device=self.device)
        loss_vgg_agg = torch.stack(vgg_losses).mean() if vgg_losses else torch.tensor(0.0, device=self.device)

        metrics['L1_Global'] = loss_l1_agg.item()
        metrics['L2_Global'] = loss_l2_agg.item()
        metrics['Pearson_Global'] = loss_pearson_agg.item()

        # 4. Auxiliary Losses (Q-Map constraints)
        loss_pd_const = losses.PD_constraint_loss(self.PD, self.otsu_mask)
        
        # TV Reg
        PDn = (self.PD / 0.70).unsqueeze(1)
        T1n = (self.T1 / 0.85).unsqueeze(1)
        T2n = (self.T2 / 0.070).unsqueeze(1)
        m_float = self.mask_brain.float()
        loss_tv = (losses.total_variation_loss_isotropic_masked(PDn, m_float) +
                   losses.total_variation_loss_isotropic_masked(T1n, m_float) +
                   losses.total_variation_loss_isotropic_masked(T2n, m_float)) / 3.0
        
        metrics['PD_Const'] = loss_pd_const.item()
        metrics['TV_Reg'] = loss_tv.item()

        # 5. Total Loss Calculation
        if optimize:
            total_loss_G = (
                loss_l1_agg * self.opt.loss_content_I_l1 +
                loss_l2_agg * self.opt.loss_content_I_l2 +
                loss_pearson_agg * self.opt.loss_content_pearson +
                loss_vgg_agg * self.opt.loss_vgg +
                loss_pd_const * self.opt.loss_PD_constraint +
                loss_tv * self.opt.loss_tv_reg 
            )
            return total_loss_G, metrics
        else:
            return 0.0, metrics
        
    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        loss_G, metrics = self.compute_losses_and_metrics(optimize=True)

        loss_G.backward()
        self.optimizer_G.step()
        return metrics

    def evaluate(self):
        """Evaluation without gradient calculation."""
        with torch.no_grad():
            self.forward()
            _, metrics = self.compute_losses_and_metrics(optimize=False)
        return metrics
    
    def warmup_optimize_parameters(self):
        """Simplified warmup."""
        self.forward()
        self.optimizer_G.zero_grad()
        
        # Simple L1 against literature values
        val_T1, val_T2, val_PD = 0.850, 0.070, 0.700
        mask = self.mask_brain.squeeze(1)
        
        loss = 0
        loss += self.criterionL1(self.PD[mask]/1, torch.full_like(self.PD[mask], val_PD)/1)
        loss += self.criterionL1(self.T1[mask]/5, torch.full_like(self.T1[mask], val_T1)/5)
        loss += self.criterionL1(self.T2[mask]/3, torch.full_like(self.T2[mask], val_T2)/3)
        
        loss.backward()
        self.optimizer_G.step()