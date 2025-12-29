# import math
# import torch
# import os
# from .base_model import BaseModel
# from . import networks
# from . import losses
# # import pytorch_msssim
# import matplotlib.pyplot as plt
# import numpy as np
# import torchio as tio
# import torch.nn.functional as F
# from Qstar.util.util import histogram_mode_from_voxels_torch_batch
# import random


# class StandardModel(BaseModel):
#     def name(self):
#         return 'StandardModel'

#     def initialize(self, opt):
#         BaseModel.initialize(self, opt)
#         self.isTrain = opt.isTrain
#         self.opt = opt

#         # Define loss names with scan-specific names.
#         if self.isTrain:
#             if self.train_phase == 'generator':
#                 self.model_names = ['G']
#                 self.loss_names = [
#                     'G_I_L1_T1w', 'G_I_L2_T1w',
#                     'G_I_L1_T2w', 'G_I_L2_T2w',
#                     'G_I_L1_FLAIR', 'G_I_L2_FLAIR',
#                     'G_I_L1', 'G_I_L2', 
#                     'G_I_pearson_T1w', 'G_I_pearson_T2w', 'G_I_pearson_FLAIR',
#                     'G_I_pearson',
#                     'PSNR_T1w', 'PSNR_T2w', 'PSNR_FLAIR',
#                     'vgg',
#                     'SSIM_T1w', 'SSIM_T2w', 'SSIM_FLAIR',
#                     'PDT1_relation', 'PD_wm', 'PD_variance', 'PD_constraint_head', 'tv_reg', 'PD_prior'
#                 ]
#             else:
#                 self.model_names = ['G', 'D1', 'D2', 'D3']
#                 self.loss_names = [
#                     'G_GAN',
#                     'G_I_L1_T1w', 'G_I_L2_T1w',
#                     'G_I_L1_T2w', 'G_I_L2_T2w',
#                     'G_I_L1_FLAIR', 'G_I_L2_FLAIR',
#                     'G_I_L1', 'G_I_L2',
#                     'G_I_pearson_T1w', 'G_I_pearson_T2w', 'G_I_pearson_FLAIR',
#                     'G_I_pearson',
#                     'D_GAN_fake', 'D_GAN_real',
#                     'PSNR_T1w', 'PSNR_T2w', 'PSNR_FLAIR',
#                     'vgg',
#                     'SSIM_T1w', 'SSIM_T2w', 'SSIM_FLAIR',
#                     'PDT1_relation', 'PD_wm', 'PD_variance', 'PD_constraint_head', 'tv_reg', 'PD_prior'
#                 ]
#         else:
#             self.model_names = ['G']
#             self.loss_names = [
#                 'G_I_L1_T1w', 'G_I_L2_T1w',
#                 'G_I_L1_T2w', 'G_I_L2_T2w',
#                 'G_I_L1_FLAIR', 'G_I_L2_FLAIR',
#                 'G_I_L1', 'G_I_L2',
#                 'G_I_pearson_T1w', 'G_I_pearson_T2w', 'G_I_pearson_FLAIR',
#                 'G_I_pearson',
#                 'PSNR_T1w', 'PSNR_T2w', 'PSNR_FLAIR',
#                 'vgg',
#                 'SSIM_T1w', 'SSIM_T2w', 'SSIM_FLAIR',
#                 'PDT1_relation', 'PD_wm', 'PD_variance', 'PD_constraint_head', 'tv_reg', 'PD_prior'
#             ]

#         # Define visuals
#         self.visual_names = ['fake_T1w', 'real_T1w', 'fake_T2w', 'real_T2w', 'fake_FLAIR', 'real_FLAIR']
#         self.visual_names.extend(['Q1', 'Q2', 'Q3'])
        
#         self.netG = networks.define_G(self.opt, opt.input_nc, opt.output_nc, 
#                                       opt.which_model_netG, opt.encoder_norm, opt.decoder_norm, opt.bottleneck_norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.input_scan_types)

#         self.criterionL1 = torch.nn.L1Loss()
#         self.criterionMSE = torch.nn.MSELoss()
#         # self.ssim_loss = pytorch_msssim.SSIM(data_range=1)
#         if opt.loss_vgg:
#             self.perceptual = losses.PerceptualLoss()
#             self.perceptual.initialize(self.criterionMSE)

#         self.rescaling_factor_bounds = opt.rescaling_factor_bounds

#         if self.isTrain:
#             if opt.optimizer == 'adam':
#                 self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
#             elif opt.optimizer == 'sgd':
#                 self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=opt.beta1, weight_decay=opt.weight_decay)
#             elif opt.optimizer == 'adamw':
#                 self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay) 

#             # self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers = [self.optimizer_G]

#     def set_epoch(self, epoch: int):
#         self._current_epoch = epoch   
            
#     # modality dropout helper
#     def _apply_modality_dropout(self, x):
#         """
#         x : tensor of shape (B, 3, H, W)
#         Returns a tensor with 0–2 channels set to zero **per sample**.
#         """
#         p = self.opt.modality_dropout_rate
#         max_missing = self.opt.modality_dropout_max_missing
#         burnin = self.opt.modality_dropout_burnin
        
#         if (not self.isTrain) or p == 0.0 or self._current_epoch < burnin:
#             return x                      # no dropout in val / test

#         B, C, _, _ = x.shape
#         device = x.device
#         keep_mask = torch.ones(B, C, 1, 1, device=device)

#         for b in range(B):
#             # decide which (if any) modalities to drop
#             drop_flags = (torch.rand(C, device=device) < p).int()
#             # never drop more than max_missing
#             if drop_flags.sum() > max_missing:
#                 # keep a random subset so that exactly max_missing are missing
#                 keep_idx = torch.randperm(C, device=device)[:C - max_missing]
#                 drop_flags[:] = 1
#                 drop_flags[keep_idx] = 0
#             # guarantee at least one modality is still present
#             if drop_flags.sum() == C:
#                 keep_flag = torch.randint(0, C, (1,), device=device)
#                 drop_flags[keep_flag] = 0
#             keep_mask[b, drop_flags.bool(), :, :] = 0

#         return x * keep_mask
    
#     def _modalities_to_list(self, keep_flags=None):
#         """
#         Build a *list* of 1-channel tensors in the order
#         given by opt.input_scan_types.
#         keep_flags: Bool tensor (len == #modalities) – if provided we
#                     skip the modalities where flag == False.
#         Returns a randomly shuffled list of the tensors to ensure
#         permutation invariance in the model.
#         """
#         tensors = []
#         for idx, scan in enumerate(self.opt.input_scan_types):
#             if keep_flags is not None and not keep_flags[idx]:
#                 continue
    
#             if scan == 'T1w':
#                 tensors.append(self.real_T1w)
#             elif scan == 'T2w':
#                 tensors.append(self.real_T2w)
#             elif scan == 'FLAIR':
#                 tensors.append(self.real_FLAIR)
#             elif scan == 'PDw' and hasattr(self, 'real_PDw'):
#                 tensors.append(self.real_PDw)
#             elif scan == 'T12w' and hasattr(self, 'real_T12w'):
#                 tensors.append(self.real_T12w)
#             elif scan == 'TI400' and hasattr(self, 'real_TI400'):
#                 tensors.append(self.real_TI400)
#             elif scan == 'DIR' and hasattr(self, 'real_DIR'):
#                 tensors.append(self.real_DIR)
#             else:
#                 raise KeyError(f'Unsupported or missing scan type: {scan}')
        
#         # Shuffle the tensors in-place to ensure permutation invariance
#         random.shuffle(tensors)
        
#         return tensors
    
#     def _choose_modalities(self):
#         """
#         Decide – once per step – which modalities will be *kept*.
#         Returns a length-M Bool tensor.
#         """
#         M        = len(self.opt.input_scan_types)
#         device   = self.device
#         p        = self.opt.modality_dropout_rate
#         max_miss = self.opt.modality_dropout_max_missing
#         burnin   = self.opt.modality_dropout_burnin

#         # keep everything while: not training, p==0, or still in burn-in
#         if (not self.isTrain) or p == 0.0 or self._current_epoch < burnin:
#             return torch.ones(M, dtype=torch.bool, device=device)

#         # 1) propose a random drop mask
#         drop = (torch.rand(M, device=device) < p)

#         # 2) respect the “max missing” limit
#         if drop.sum() > max_miss:
#             keep_idx = torch.randperm(M, device=device)[: M - max_miss]
#             drop[:] = True
#             drop[keep_idx] = False

#         # 3) never drop *all* modalities
#         if drop.all():
#             drop[torch.randint(0, M, (1,), device=device)] = False

#         return ~drop        # keep-flags (True ⇔ fed to the model)
            
#     def set_input(self, input, phase_input='train'):
#         self.mask_brain = input['brain_mask'][tio.DATA].to(self.device).squeeze(-1).bool()
#         self.otsu_mask = input['otsu_mask'][tio.DATA].to(self.device).squeeze(-1).bool()
#         # self.pd_otsu_mask = input['pd_otsu_mask'][tio.DATA].to(self.device).squeeze(-1).bool()
#         self.real_T1w = input['T1w'][tio.DATA].to(dtype=torch.float, device=self.device).squeeze(-1) * self.mask_brain
#         self.real_T2w = input['T2w'][tio.DATA].to(dtype=torch.float, device=self.device).squeeze(-1) * self.mask_brain
#         self.real_FLAIR = input['FLAIR'][tio.DATA].to(dtype=torch.float, device=self.device).squeeze(-1) * self.mask_brain
    
#         # Map scan types to their corresponding tensors
#         scan_map = {
#             'T1w': self.real_T1w,
#             'T2w': self.real_T2w,
#             'FLAIR': self.real_FLAIR
#         }
        
#         # Only include the selected modalities
#         selected_scans = []
#         for scan_name in self.opt.input_scan_types:
#             if scan_name not in scan_map:
#                 raise KeyError(f"Requested '{scan_name}' in input_scan_types, but it isn't in scan_map.")
#             selected_scans.append(scan_map[scan_name])
            
#         if 'shared_attn_unet' in self.opt.which_model_netG:
#             keep_flags = self._choose_modalities()          # length-M Bool
#             self.G_inputs = self._modalities_to_list(keep_flags)


#         # Sequence parameters
#         self.T1w_TR = input['T1w_TR'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.T1w_TE = input['T1w_TE'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.T1w_TI = input['T1w_TI'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.T1w_FA = input['T1w_FA'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.T2w_TR = input['T2w_TR'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.T2w_TE = input['T2w_TE'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.T2w_FA = input['T2w_FA'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.FLAIR_TR = input['FLAIR_TR'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.FLAIR_TE = input['FLAIR_TE'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.FLAIR_TI = input['FLAIR_TI'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.FLAIR_FA = input['FLAIR_FA'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.T1w_rescaling_factor = input['T1w_rescaling_factor'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.T2w_rescaling_factor = input['T2w_rescaling_factor'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
#         self.FLAIR_rescaling_factor = input['FLAIR_rescaling_factor'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        
#         # sequence types
#         self.T1w_seq_type = input['T1w_seq_type']
#         self.T2w_seq_type = input['T2w_seq_type']
#         self.FLAIR_seq_type = input['FLAIR_seq_type']
           
#     def _batch_signal_model(
#             self,
#             PD, T1, T2,
#             TR, TE, TI, FA,
#             seq_types,
#             TI2=None):
#         """
#         Apply `self.signal_model` one sample at a time so that every
#         element can have its *own* `seq_type`.

#         tensors: B×H×W (no channel dim, you already stripped it above)
#         seq_types:  list  (length == B)  or  1-D tensor of dtype=object/str
#         """
#         outs = []
#         B = PD.size(0)
#         for b in range(B):
#             out = self.signal_model(
#                 PD[b], T1[b], T2[b],
#                 TR[b], TE[b],
#                 None if TI  is None else TI[b],
#                 FA[b],
#                 seq_type=str(seq_types[b]),          
#                 TI2=None if TI2 is None else TI2[b],
#             )
#             out = out.squeeze(0)
#             outs.append(out)
#         return torch.stack(outs, dim=0)              # B×1×H×W

#     def signal_model(self, PD, T1, T2, TR, TE, TI, FA, seq_type='TFE', TI2=None):
#         TR = TR / 1000
#         TE = TE / 1000
#         if TI is not None:
#             TI = TI / 1000
#         if TI2 is not None:
#             TI2 = TI2 / 1000
#         TR = TR.unsqueeze(-1)
#         TE = TE.unsqueeze(-1)
#         if TI is not None:
#             TI = TI.unsqueeze(-1)
#         if TI2 is not None:
#             TI2 = TI2.unsqueeze(-1)
#         FA = FA.unsqueeze(-1)
#         FA = torch.deg2rad(FA)
        
#         if seq_type == 'MPRAGE':
#             out = torch.abs(PD * (1 - 2 * torch.exp(-TI / (T1 + 1e-6)) + torch.exp(-TR / (T1 + 1e-6))) * torch.sin(FA) * torch.exp(-TE / (T2 + 1e-6)) / (1 + torch.cos(FA) * torch.exp(-TR / (T1 + 1e-6))))
#         elif seq_type == 'TFE':
#             out = PD * torch.sin(FA) * (1 - torch.exp(-TR / (T1+1e-6))) * torch.exp(-TE / (T2+1e-6)) / (1 - torch.cos(FA) * torch.exp(-TR / (T1+1e-6)))
#         elif seq_type == 'SE':
#             out =  PD * torch.sin(FA) * (1 - torch.exp(-TR / (T1+1e-6))) * torch.exp(-TE / (T2+1e-6)) / (1 - torch.cos(FA) * torch.exp(-TR / (T1+1e-6)))
#         elif seq_type == 'TSE':
#             out = PD * torch.exp(-TE / (T2 + 1e-6)) * (1 - torch.exp(-TR / (T1 + 1e-6)))
#         elif seq_type == 'IR' or seq_type == 'FLAIR':
#             out = torch.abs(PD * torch.exp(-TE / (T2 + 1e-6)) * (1 - 2 * torch.exp(-TI / (T1 + 1e-6)) + torch.exp(-TR / (T1 + 1e-6))))
#         elif seq_type == 'T12w_TSE':
#             out = PD * torch.exp(-TE / (T2 + 1e-6)) * (1 - torch.exp(-TR / (T1 + 1e-6))) / (1 - torch.cos(FA) * torch.exp(-TR / (T1 + 1e-6)))
#         elif seq_type == 'DIR':
#             out = torch.abs(PD * torch.exp(-TE / (T2 + 1e-6)) * (1 - 2 * torch.exp(-TI / (T1 + 1e-6)) + 2 * torch.exp(-TI2 / (T1 + 1e-6)) - torch.exp(-TR / (T1 + 1e-6))))
#         else:
#             raise ValueError('Seq type  not recognized')
#         if out.dim() == 2:              
#             out = out.unsqueeze(0)      
#         return out[:, None, :, :]       
    

#     def forward(self, is_train=True):
#         if self.opt.multi_protocol_conditioning:
#             if 'multiencoder' not in self.opt.which_model_netG:
#                 self.fake_B = self.netG(self.real_A, self.conditioning_vector)
#             else:
#                 # Create inputs and conditioning vectors lists based on selected scan types
#                 inputs = []
#                 conditioning_vectors = []
#                 for scan in self.opt.input_scan_types:
#                     if scan == 'T1w':
#                         inputs.append(self.real_T1w)
#                         conditioning_vectors.append(self.conditioning_vector_T1w)
#                     elif scan == 'T2w':
#                         inputs.append(self.real_T2w)
#                         conditioning_vectors.append(self.conditioning_vector_T2w)
#                     elif scan == 'FLAIR':
#                         inputs.append(self.real_FLAIR)
#                         conditioning_vectors.append(self.conditioning_vector_FLAIR)
#                 self.fake_B = self.netG(*inputs, *conditioning_vectors)
#         else:
#             if 'shared_attn_unet' in self.opt.which_model_netG:
#                 self.fake_B = self.netG(self.G_inputs) 
#             elif 'multiencoder' not in self.opt.which_model_netG:
#                 self.fake_B = self.netG(self.real_A)
                
#             else: # multiple encoder branches
#             # Create inputs list based on selected scan types 
#                 inputs = []
#                 for scan in self.opt.input_scan_types:
#                     if scan == 'T1w':
#                         inputs.append(self.real_T1w)
#                     elif scan == 'T2w':
#                         inputs.append(self.real_T2w)
#                     elif scan == 'FLAIR':
#                         inputs.append(self.real_FLAIR)
#                 self.fake_B = self.netG(*inputs)
            
#         # For display purposes
#         self.Q1 = (self.fake_B[:, 0, :, :]*1)[:, None, :, :] * self.mask_brain  # PD
#         self.Q2 = (self.fake_B[:, 1, :, :]*5)[:, None, :, :] * self.mask_brain  # T1 (scaled)
#         self.Q3 = (self.fake_B[:, 2, :, :]*3)[:, None, :, :] * self.mask_brain  # T2 (scaled)
        
#         # For further processing:
#         self.PD = (self.fake_B[:, 0, :, :]*1) * self.mask_brain.squeeze(1)
#         self.T1 = (self.fake_B[:, 1, :, :]*5) * self.mask_brain.squeeze(1)
#         self.T2 = (self.fake_B[:, 2, :, :]*3) * self.mask_brain.squeeze(1)
        
#         self.fake_T1w  = self._batch_signal_model(self.PD, self.T1, self.T2, self.T1w_TR, self.T1w_TE, self.T1w_TI, self.T1w_FA, self.T1w_seq_type)                                   
#         self.fake_T2w  = self._batch_signal_model(self.PD, self.T1, self.T2, self.T2w_TR, self.T2w_TE, None,  self.T2w_FA, self.T2w_seq_type)
#         self.fake_FLAIR = self._batch_signal_model(self.PD, self.T1, self.T2, self.FLAIR_TR, self.FLAIR_TE, self.FLAIR_TI, self.FLAIR_FA, self.FLAIR_seq_type)
      
#         self.loss_PD_constraint_head = losses.PD_constraint_loss(self.PD, self.otsu_mask)
        
#         PDn = (self.PD / 0.70).unsqueeze(1)
#         T1n = (self.T1 / 0.85).unsqueeze(1)   # seconds
#         T2n = (self.T2 / 0.070).unsqueeze(1)  # seconds
#         m   = self.mask_brain.float()
#         self.loss_tv_reg = (
#             losses.total_variation_loss_isotropic_masked(PDn, m) +
#             losses.total_variation_loss_isotropic_masked(T1n, m) +
#             losses.total_variation_loss_isotropic_masked(T2n, m)
#         ) / 3.0

        
        
#     def backward_G(self):
#         # Initialize the overall generator loss
#         self.loss_G = 0

#         # Define global rescaling bounds 
#         bound_percentage = self.rescaling_factor_bounds
#         global_rescaling_T1w_bounds = (
#             self.T1w_rescaling_factor - bound_percentage * self.T1w_rescaling_factor,
#             self.T1w_rescaling_factor + bound_percentage * self.T1w_rescaling_factor
#         )
#         global_rescaling_T2w_bounds = (
#             self.T2w_rescaling_factor - bound_percentage * self.T2w_rescaling_factor,
#             self.T2w_rescaling_factor + bound_percentage * self.T2w_rescaling_factor
#         )
#         global_rescaling_FLAIR_bounds = (
#             self.FLAIR_rescaling_factor - bound_percentage * self.FLAIR_rescaling_factor,
#             self.FLAIR_rescaling_factor + bound_percentage * self.FLAIR_rescaling_factor
#         )

#         # Prepare a list with modality-specific info
#         scan_info = [
#             ('T1w', self.fake_T1w, self.real_T1w, self.netD1, global_rescaling_T1w_bounds, self.T1w_rescaling_factor, 'rescaling_factor_T1w'),
#             ('T2w', self.fake_T2w, self.real_T2w, self.netD2, global_rescaling_T2w_bounds, self.T2w_rescaling_factor, 'rescaling_factor_T2w'),
#             ('FLAIR', self.fake_FLAIR, self.real_FLAIR, self.netD3, global_rescaling_FLAIR_bounds, self.FLAIR_rescaling_factor, 'rescaling_factor_FLAIR')
#         ]

#         normalized_losses_l1 = []
#         normalized_losses_l2 = []
#         pearson_losses = []
#         vgg_losses = []
        
#         # Initialize losses for each scan type
#         self.loss_G_I_L1_T1w = self.loss_G_I_L2_T1w = 0
#         self.loss_G_I_L1_T2w = self.loss_G_I_L2_T2w = 0
#         self.loss_G_I_L1_FLAIR = self.loss_G_I_L2_FLAIR = 0
#         self.loss_G_I_pearson_T1w = self.loss_G_I_pearson_T2w = self.loss_G_I_pearson_FLAIR = 0
#         self.loss_G_I_pearson = 0
#         self.loss_G_I_L1 = self.loss_G_I_L2 = 0
#         self.loss_SSIM_T1w = self.loss_SSIM_T2w = self.loss_SSIM_FLAIR = 0
#         self.loss_PSNR_T1w = self.loss_PSNR_T2w = self.loss_PSNR_FLAIR = 0
#         self.loss_vgg = 0
        
#         for scan, fake, real, netD, bounds, exact_rescaling_factor, rescaling_attr in scan_info:
#             # For GAN loss on generator (if in 'together' phase), only add for selected modalities.
#             if self.isTrain and self.train_phase == 'together' and scan in self.opt.input_scan_types:
#                 fake_AB = torch.cat((self.real_A, fake), 1)
#                 pred_fake = netD(fake_AB)
#                 loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.loss_GAN
#                 self.loss_G += loss_G_GAN  # add GAN loss only for selected scans

#             if self.opt.rescaling_method == 'mean':
#                 # Compute masked means for rescaling
#                 mask = self.mask_gm_wm.float()
#                 masked_sum_fake = (fake * mask).sum(dim=(1, 2, 3))
#                 masked_sum_real = (real * mask).sum(dim=(1, 2, 3))
#                 mask_count = mask.sum(dim=(1, 2, 3))
#                 mean_fake = masked_sum_fake / (mask_count + 1e-6)
#                 mean_real = masked_sum_real / (mask_count + 1e-6)

#                 # Compute and clamp the rescaling factor
#                 rescaling_factor = mean_real / (mean_fake + 1e-6)
#                 lower_bound, upper_bound = bounds
#                 lower_bound = lower_bound.squeeze(1)
#                 upper_bound = upper_bound.squeeze(1)
#                 rescaling_factor = torch.clamp(rescaling_factor, min=lower_bound, max=upper_bound)
#                 rescaling_factor = rescaling_factor.view(-1, 1, 1, 1)
#                 setattr(self, rescaling_attr, rescaling_factor)
#             elif self.opt.rescaling_method == 'mode':
#                 masked_real = real * self.mask_brain
#                 masked_fake = fake * self.mask_brain
#                 # Compute the mode of the real and fake images
#                 if 'UPENN-GBM' in self.opt.dataroot:
#                     if scan == 'T1w':
#                         percentile_threshold_lower = 1
#                         percentile_threshold_upper = 99
#                     elif scan == 'T2w':
#                         percentile_threshold_lower = 1
#                         percentile_threshold_upper = 99
#                     elif scan == 'FLAIR':
#                         percentile_threshold_lower = 1
#                         percentile_threshold_upper = 99
#                 elif 'UMCU' in self.opt.dataroot:
#                     if scan == 'T1w':
#                         percentile_threshold_lower = 50
#                         percentile_threshold_upper = 95
#                     elif scan == 'T2w':
#                         percentile_threshold_lower = 25
#                         percentile_threshold_upper = 75
#                     elif scan == 'FLAIR':
#                         percentile_threshold_lower = 50
#                         percentile_threshold_upper = 95
                
#                 mode_real = histogram_mode_from_voxels_torch_batch(masked_real,
#                                                                     percentile_threshold_lower=percentile_threshold_lower,
#                                                                     percentile_threshold_upper=percentile_threshold_upper)
#                 mode_fake = histogram_mode_from_voxels_torch_batch(masked_fake, 
#                                                                     percentile_threshold_lower=percentile_threshold_lower,
#                                                                     percentile_threshold_upper=percentile_threshold_upper)
#                 # Compute the rescaling factor and clamp
#                 rescaling_factor = mode_real / (mode_fake + 1e-6)
#                 lower_bound, upper_bound = bounds
#                 lower_bound = lower_bound.squeeze(1)
#                 upper_bound = upper_bound.squeeze(1)
#                 rescaling_factor = torch.clamp(rescaling_factor, min=lower_bound, max=upper_bound)
#                 rescaling_factor = rescaling_factor.view(-1, 1, 1, 1)
#                 setattr(self, rescaling_attr, rescaling_factor)
#             elif self.opt.rescaling_method == 'fixed_per_subject':
#                 rescaling_factor = exact_rescaling_factor
#                 rescaling_factor = rescaling_factor.view(-1, 1, 1, 1)
#                 setattr(self, rescaling_attr, rescaling_factor)
#             elif self.opt.rescaling_method == 'closed_form_least_squares':
#                 # flatten spatial dims
#                 fake_flat = fake.view(fake.size(0), -1)
#                 real_flat = real.view(real.size(0), -1)
#                 # mask_flat = self.otsu_mask.view(self.otsu_mask.size(0), -1)
#                 mask_flat = self.mask_brain.view(self.mask_brain.size(0), -1)
#                 # compute numerator and denominator per sample
#                 num = (mask_flat * fake_flat * real_flat).sum(dim=1)
#                 den = (mask_flat * fake_flat * fake_flat).sum(dim=1).clamp(min=1e-6)
#                 a = (num / den).detach()    # (B,)
#                 # reshape for broadcasting
#                 rescaling_factor = a.view(-1, 1, 1, 1)
#                 setattr(self, rescaling_attr, rescaling_factor)
#             elif self.opt.rescaling_method == 'closed_form_least_squares_bounded':
#                 # flatten spatial dims
#                 fake_flat = fake.view(fake.size(0), -1)
#                 real_flat = real.view(real.size(0), -1)
#                 # mask_flat = self.otsu_mask.view(self.otsu_mask.size(0), -1)
#                 mask_flat = self.mask_brain.view(self.mask_brain.size(0), -1)
#                 # compute numerator and denominator per sample
#                 num = (mask_flat * fake_flat * real_flat).sum(dim=1)
#                 den = (mask_flat * fake_flat * fake_flat).sum(dim=1).clamp(min=1e-6)
#                 a = num / den    # (B,)
#                 # clamp within bounds
#                 lower, upper = bounds
#                 lower = lower.squeeze(1); upper = upper.squeeze(1)
#                 a = torch.min(torch.max(a, lower), upper)
#                 # reshape for broadcasting
#                 rescaling_factor = a.view(-1, 1, 1, 1)
#                 setattr(self, rescaling_attr, rescaling_factor)
#             elif self.opt.rescaling_method == 'learned_gain':
#                 raise NotImplementedError('Learned gain rescaling method is not implemented yet')
#             else:
#                 raise ValueError('Rescaling method not recognized')

#             # Apply rescaling
#             fake_rescaled = fake * rescaling_factor
#             # Update the fake image attribute
#             if rescaling_attr == 'rescaling_factor_T1w':
#                 self.fake_T1w = fake_rescaled
#             elif rescaling_attr == 'rescaling_factor_T2w':
#                 self.fake_T2w = fake_rescaled
#             elif rescaling_attr == 'rescaling_factor_FLAIR':
#                 self.fake_FLAIR = fake_rescaled


#             L1_loss = self.criterionL1(fake_rescaled, real)
#             L2_loss = self.criterionMSE(fake_rescaled, real)
            
#             pearson_loss = losses.compute_pearson_loss(fake, real)
                

#             if self.opt.use_normalized_losses:
#                 # Aggregate mean_real over the batch to get a single scalar value
#                 global_mean_real = mean_real.mean()
#                 normalized_loss_L1 = L1_loss / (global_mean_real + 1e-6)
#                 normalized_loss_L2 = L2_loss / (global_mean_real + 1e-6)
#             else:
#                 normalized_loss_L1 = L1_loss
#                 normalized_loss_L2 = L2_loss

#             # Always store metrics (even for modalities not used in training)
#             setattr(self, f'loss_G_I_L1_{scan}', normalized_loss_L1)
#             setattr(self, f'loss_G_I_L2_{scan}', normalized_loss_L2)
#             setattr(self, f'loss_G_I_pearson_{scan}', pearson_loss)
            
#             # Only add content losses to backprop if this scan is used for training
#             if scan in self.opt.input_scan_types:
#                 normalized_losses_l1.append(normalized_loss_L1)
#                 normalized_losses_l2.append(normalized_loss_L2)
#                 # Store the losses for Pearson correlation
#                 pearson_losses.append(pearson_loss)
                
#             if self.opt.loss_vgg and scan in self.opt.input_scan_types:
#                 # Get rescaled fake image
#                 fake_rescaled = fake * rescaling_factor
#                 # Calculate perceptual loss between rescaled fake and real
#                 perceptual_loss = self.perceptual.get_loss(fake_rescaled, real)
#                 vgg_losses.append(perceptual_loss)
                
#             # Compute additional metrics (SSIM and PSNR) regardless of training selection
#             if self.opt.calc_additional_metrics:
#                 with torch.no_grad():
#                     ssim_val = losses.SSIM_crop(torch.clone(real), torch.clone(fake_rescaled), self.mask_brain)
#                     psnr_val = losses.PSNR(torch.clone(real), torch.clone(fake_rescaled), self.mask_brain)
#                     setattr(self, f'loss_SSIM_{scan}', ssim_val)
#                     setattr(self, f'loss_PSNR_{scan}', psnr_val)

#         # If any selected modality losses were accumulated, combine them
#         if normalized_losses_l1 or normalized_losses_l2:
#             self.loss_G_I_L1 = torch.stack(normalized_losses_l1).mean()
#             self.loss_G_I_L2 = torch.stack(normalized_losses_l2).mean()
#         else:
#             self.loss_G_I_L1 = 0
#             self.loss_G_I_L2 = 0
#         setattr(self, f'loss_G_I_L1', self.loss_G_I_L1)
#         setattr(self, f'loss_G_I_L2', self.loss_G_I_L2)
        
#         # Add pearson loss if calculated
#         if pearson_losses:
#             self.loss_pearson = torch.stack(pearson_losses).mean()
#         else:
#             self.loss_pearson = torch.tensor(0.0).to(self.device)
#         setattr(self, 'loss_G_I_pearson', self.loss_pearson)    
        
        
#         # If any VGG losses were calculated, combine them
#         if self.opt.loss_vgg and vgg_losses:
#             self.loss_vgg = torch.stack(vgg_losses).mean()
#         else:
#             self.loss_vgg = torch.tensor(0.0).to(self.device)

#         # Add the content losses (weighted) to the overall generator loss
#         self.loss_G += self.loss_G_I_L1 * self.opt.loss_content_I_l1
#         self.loss_G += self.loss_G_I_L2 * self.opt.loss_content_I_l2
#         self.loss_G += self.loss_pearson * self.opt.loss_content_pearson

#         # Add any additional losses (e.g., PDT1 relation, PD wm, PD variance, perceptual) as before
#         self.loss_G += self.loss_PDT1_relation * self.opt.loss_PDT1_relation
#         self.loss_G += self.loss_PD_wm * self.opt.loss_PD_wm
#         self.loss_G += self.loss_PD_variance * self.opt.loss_PD_variance
#         self.loss_G += self.loss_PD_constraint_head * self.opt.loss_PD_constraint
#         self.loss_G += self.loss_vgg * self.opt.loss_vgg
#         self.loss_G += self.loss_tv_reg * self.opt.loss_tv_reg
#         self.loss_G += self.opt.loss_PD_prior * self.loss_PD_prior

#         # Check for NaN values
#         if math.isnan(self.loss_G_I_L1.item()):
#             raise ValueError('Training Loss L1 is nan')
#         if math.isnan(self.loss_G_I_L2.item()):
#             raise ValueError('Training Loss L2 is nan')
        
#         self.loss_G.backward()

#     def get_val_losses(self):
#         """
#         Compute validation losses and metrics with the same loop structure as backward_G,
#         but without performing backpropagation.
#         """
#         # Initialize the overall generator loss
#         self.loss_G = 0

#         # Define global rescaling bounds 
#         bound_percentage = self.rescaling_factor_bounds
#         global_rescaling_T1w_bounds = (
#             self.T1w_rescaling_factor - bound_percentage * self.T1w_rescaling_factor,
#             self.T1w_rescaling_factor + bound_percentage * self.T1w_rescaling_factor
#         )
#         global_rescaling_T2w_bounds = (
#             self.T2w_rescaling_factor - bound_percentage * self.T2w_rescaling_factor,
#             self.T2w_rescaling_factor + bound_percentage * self.T2w_rescaling_factor
#         )
#         global_rescaling_FLAIR_bounds = (
#             self.FLAIR_rescaling_factor - bound_percentage * self.FLAIR_rescaling_factor,
#             self.FLAIR_rescaling_factor + bound_percentage * self.FLAIR_rescaling_factor
#         )

#         # Prepare a list with modality-specific info
#         # Each tuple contains:
#         # (scan identifier, fake image, real image, corresponding discriminator, bounds,
#         #  exact rescaling factor (for fixed method), attribute name to store the rescaling factor)
#         scan_info = [
#             ('T1w', self.fake_T1w, self.real_T1w, self.netD1, global_rescaling_T1w_bounds, self.T1w_rescaling_factor, 'rescaling_factor_T1w'),
#             ('T2w', self.fake_T2w, self.real_T2w, self.netD2, global_rescaling_T2w_bounds, self.T2w_rescaling_factor, 'rescaling_factor_T2w'),
#             ('FLAIR', self.fake_FLAIR, self.real_FLAIR, self.netD3, global_rescaling_FLAIR_bounds, self.FLAIR_rescaling_factor, 'rescaling_factor_FLAIR')
#         ]

#         normalized_losses_l1 = []
#         normalized_losses_l2 = []
#         pearson_losses = []
#         vgg_losses = []
        
#         # Initialize losses for each scan type
#         self.loss_G_I_L1_T1w = self.loss_G_I_L2_T1w = 0
#         self.loss_G_I_L1_T2w = self.loss_G_I_L2_T2w = 0
#         self.loss_G_I_L1_FLAIR = self.loss_G_I_L2_FLAIR = 0
#         self.loss_G_I_pearson_T1w = self.loss_G_I_pearson_T2w = self.loss_G_I_pearson_FLAIR = 0
#         self.loss_G_I_pearson = 0
#         self.loss_G_I_L1 = self.loss_G_I_L2 = 0
#         self.loss_SSIM_T1w = self.loss_SSIM_T2w = self.loss_SSIM_FLAIR = 0
#         self.loss_PSNR_T1w = self.loss_PSNR_T2w = self.loss_PSNR_FLAIR = 0
#         self.loss_vgg = 0

#         for scan, fake, real, netD, bounds, exact_rescaling_factor, rescaling_attr in scan_info:
#             # For GAN loss on generator (if in 'together' phase), only add for selected modalities.
#             if self.isTrain and self.train_phase == 'together' and scan in self.opt.input_scan_types:
#                 fake_AB = torch.cat((self.real_A, fake), 1)
#                 pred_fake = netD(fake_AB)
#                 loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.loss_GAN
#                 self.loss_G += loss_G_GAN  # add GAN loss only for selected scans

#             # Compute rescaling factor based on the chosen method
#             if self.opt.rescaling_method == 'mean':
#                 mask = self.mask_gm_wm.float()
#                 masked_sum_fake = (fake * mask).sum(dim=(1, 2, 3))
#                 masked_sum_real = (real * mask).sum(dim=(1, 2, 3))
#                 mask_count = mask.sum(dim=(1, 2, 3))
#                 mean_fake = masked_sum_fake / (mask_count + 1e-6)
#                 mean_real = masked_sum_real / (mask_count + 1e-6)
#                 rescaling_factor = mean_real / (mean_fake + 1e-6)
#                 lower_bound, upper_bound = bounds
#                 lower_bound = lower_bound.squeeze(1)
#                 upper_bound = upper_bound.squeeze(1)
#                 rescaling_factor = torch.clamp(rescaling_factor, min=lower_bound, max=upper_bound)
#                 rescaling_factor = rescaling_factor.view(-1, 1, 1, 1)
#                 setattr(self, rescaling_attr, rescaling_factor)
#             elif self.opt.rescaling_method == 'mode':
#                 masked_real = real * self.mask_brain
#                 masked_fake = fake * self.mask_brain
#                 if 'UPENN-GBM' in self.opt.dataroot:
#                     if scan == 'T1w' or scan == 'T2w' or scan == 'FLAIR':
#                         percentile_threshold_lower = 1
#                         percentile_threshold_upper = 99
#                 elif 'UMCU' in self.opt.dataroot:
#                     if scan == 'T1w':
#                         percentile_threshold_lower = 50
#                         percentile_threshold_upper = 95
#                     elif scan == 'T2w':
#                         percentile_threshold_lower = 25
#                         percentile_threshold_upper = 75
#                     elif scan == 'FLAIR':
#                         percentile_threshold_lower = 50
#                         percentile_threshold_upper = 95
#                 mode_real = histogram_mode_from_voxels_torch_batch(masked_real,
#                                                                 percentile_threshold_lower=percentile_threshold_lower,
#                                                                 percentile_threshold_upper=percentile_threshold_upper)
#                 mode_fake = histogram_mode_from_voxels_torch_batch(masked_fake,
#                                                                 percentile_threshold_lower=percentile_threshold_lower,
#                                                                 percentile_threshold_upper=percentile_threshold_upper)
#                 rescaling_factor = mode_real / (mode_fake + 1e-6)
#                 lower_bound, upper_bound = bounds
#                 lower_bound = lower_bound.squeeze(1)
#                 upper_bound = upper_bound.squeeze(1)
#                 rescaling_factor = torch.clamp(rescaling_factor, min=lower_bound, max=upper_bound)
#                 rescaling_factor = rescaling_factor.view(-1, 1, 1, 1)
#                 setattr(self, rescaling_attr, rescaling_factor)
#             elif self.opt.rescaling_method == 'fixed_per_subject':
#                 rescaling_factor = exact_rescaling_factor
#                 rescaling_factor = rescaling_factor.view(-1, 1, 1, 1)
#                 setattr(self, rescaling_attr, rescaling_factor)
#             elif self.opt.rescaling_method == 'closed_form_least_squares':
#                 # flatten spatial dims
#                 fake_flat = fake.view(fake.size(0), -1)
#                 real_flat = real.view(real.size(0), -1)
#                 # mask_flat = self.otsu_mask.view(self.otsu_mask.size(0), -1)
#                 mask_flat = self.mask_brain.view(self.mask_brain.size(0), -1)
#                 # compute numerator and denominator per sample
#                 num = (mask_flat * fake_flat * real_flat).sum(dim=1)
#                 den = (mask_flat * fake_flat * fake_flat).sum(dim=1).clamp(min=1e-6)
#                 a = (num / den).detach()    # (B,)
#                 # # clamp within bounds
#                 # lower, upper = bounds
#                 # lower = lower.squeeze(1); upper = upper.squeeze(1)
#                 # a = torch.min(torch.max(a, lower), upper)
#                 # reshape for broadcasting
#                 rescaling_factor = a.view(-1, 1, 1, 1)
#                 setattr(self, rescaling_attr, rescaling_factor)
#             elif self.opt.rescaling_method == 'closed_form_least_squares_bounded':
#                 # flatten spatial dims
#                 fake_flat = fake.view(fake.size(0), -1)
#                 real_flat = real.view(real.size(0), -1)
#                 # mask_flat = self.otsu_mask.view(self.otsu_mask.size(0), -1)
#                 mask_flat = self.mask_brain.view(self.mask_brain.size(0), -1)
#                 # compute numerator and denominator per sample
#                 num = (mask_flat * fake_flat * real_flat).sum(dim=1)
#                 den = (mask_flat * fake_flat * fake_flat).sum(dim=1).clamp(min=1e-6)
#                 a = num / den    # (B,)
#                 # clamp within bounds
#                 lower, upper = bounds
#                 lower = lower.squeeze(1); upper = upper.squeeze(1)
#                 a = torch.min(torch.max(a, lower), upper)
#                 # reshape for broadcasting
#                 rescaling_factor = a.view(-1, 1, 1, 1)
#                 setattr(self, rescaling_attr, rescaling_factor)
#             elif self.opt.rescaling_method == 'learned_gain':
#                 raise NotImplementedError('Learned gain rescaling method is not implemented yet')
#             else:
#                 raise ValueError('Rescaling method not recognized')

#             # Apply rescaling and update the fake image attribute for the current scan
#             fake_rescaled = fake * rescaling_factor
#             if rescaling_attr == 'rescaling_factor_T1w':
#                 self.fake_T1w = fake_rescaled
#             elif rescaling_attr == 'rescaling_factor_T2w':
#                 self.fake_T2w = fake_rescaled
#             elif rescaling_attr == 'rescaling_factor_FLAIR':
#                 self.fake_FLAIR = fake_rescaled

#             L1_loss = self.criterionL1(fake_rescaled, real)
#             L2_loss = self.criterionMSE(fake_rescaled, real)
#             pearson_loss = losses.compute_pearson_loss(fake, real)
            
#             if self.opt.use_normalized_losses:
#                 global_mean_real = mean_real.mean()  # Use mean over batch for normalization
#                 normalized_loss_L1 = L1_loss / (global_mean_real + 1e-6)
#                 normalized_loss_L2 = L2_loss / (global_mean_real + 1e-6)
#             else:
#                 normalized_loss_L1 = L1_loss
#                 normalized_loss_L2 = L2_loss

#             setattr(self, f'loss_G_I_L1_{scan}', normalized_loss_L1)
#             setattr(self, f'loss_G_I_L2_{scan}', normalized_loss_L2)
#             setattr(self, f'loss_G_I_pearson_{scan}', pearson_loss if self.opt.loss_content_pearson else torch.tensor(0.0).to(self.device))
            
#             # Add pearson loss if calculated
#             if pearson_losses:
#                 self.loss_pearson = torch.stack(pearson_losses).mean()
#             else:
#                 self.loss_pearson = torch.tensor(0.0).to(self.device)
#             setattr(self, 'loss_G_I_pearson', self.loss_pearson)   

#             if scan in self.opt.input_scan_types:
#                 normalized_losses_l1.append(normalized_loss_L1)
#                 normalized_losses_l2.append(normalized_loss_L2)
#                 pearson_losses.append(pearson_loss)
                
#             if self.opt.loss_vgg and scan in self.opt.input_scan_types:
#                 # Get rescaled fake image
#                 fake_rescaled = fake * rescaling_factor
#                 # Calculate perceptual loss between rescaled fake and real
#                 perceptual_loss = self.perceptual.get_loss(fake_rescaled, real)
#                 vgg_losses.append(perceptual_loss)
#                 self.loss_vgg = torch.stack(vgg_losses).mean()
#             else:
#                 self.loss_vgg = torch.tensor(0.0).to(self.device)

#             # Compute additional metrics (SSIM and PSNR)
#             if self.opt.calc_additional_metrics:
#                     ssim_val = losses.SSIM_crop(torch.clone(real), torch.clone(fake_rescaled), self.mask_brain)
#                     psnr_val = losses.PSNR(torch.clone(real), torch.clone(fake_rescaled), self.mask_brain)
#                     setattr(self, f'loss_SSIM_{scan}', ssim_val)
#                     setattr(self, f'loss_PSNR_{scan}', psnr_val)

#         # Combine content losses for selected modalities
#         if normalized_losses_l1 or normalized_losses_l2:
#             self.loss_G_I_L1 = torch.stack(normalized_losses_l1).mean()
#             self.loss_G_I_L2 = torch.stack(normalized_losses_l2).mean()
#         else:
#             self.loss_G_I_L1 = 0
#             self.loss_G_I_L2 = 0
#         setattr(self, f'loss_G_I_L1', self.loss_G_I_L1)
#         setattr(self, f'loss_G_I_L2', self.loss_G_I_L2)
        
#         # Add pearson loss if calculated
#         if self.opt.loss_content_pearson and pearson_losses:
#             self.loss_pearson = torch.stack(pearson_losses).mean()
#         else:
#             self.loss_pearson = torch.tensor(0.0).to(self.device)
#         setattr(self, 'loss_G_I_pearson', self.loss_pearson)   

        
#     def optimize_parameters(self):
#         if self.isTrain and self.train_phase == 'together':
#             self.forward()
#             # Only update discriminators for the modalities selected for training.
#             if 'T1w' in self.opt.input_scan_types:
#                 self.set_requires_grad(self.netD1, True)
#             if 'T2w' in self.opt.input_scan_types:
#                 self.set_requires_grad(self.netD2, True)
#             if 'FLAIR' in self.opt.input_scan_types:
#                 self.set_requires_grad(self.netD3, True)
                
#             for iter_d in range(self.disc_step):
#                 if 'T1w' in self.opt.input_scan_types:
#                     self.optimizer_D1.zero_grad()
#                     self.backward_D1()
#                     self.optimizer_D1.step()

#                 if 'T2w' in self.opt.input_scan_types:
#                     self.optimizer_D2.zero_grad()
#                     self.backward_D2()
#                     self.optimizer_D2.step()

#                 if 'FLAIR' in self.opt.input_scan_types:
#                     self.optimizer_D3.zero_grad()
#                     self.backward_D3()
#                     self.optimizer_D3.step()

#             if 'T1w' in self.opt.input_scan_types:
#                 self.set_requires_grad(self.netD1, False)
#             if 'T2w' in self.opt.input_scan_types:
#                 self.set_requires_grad(self.netD2, False)
#             if 'FLAIR' in self.opt.input_scan_types:
#                 self.set_requires_grad(self.netD3, False)
                
#             self.optimizer_G.zero_grad()
#             self.backward_G()
#             self.optimizer_G.step()
#         else:
#             self.forward()
#             self.optimizer_G.zero_grad()
#             self.backward_G()
#             self.optimizer_G.step()
            
            
#     def warmup_backward_G(self):
#         '''Idea is to warmup model on first epoch. Supervised learning on generated qmaps.
#             GT is literature WM values in whole brain mask for PD, T1 and T2
#         '''
#         # get GT_prime. By getting brain mask and filling it with literature values.
#         value_T1 = 0.850
#         value_T2 = 0.070
#         value_PD = 0.700
        
#         T1_GT_prime = torch.zeros_like(self.PD)  # Initialize with zeros
#         T1_GT_prime[self.mask_brain.squeeze(1)] = value_T1  # Fill brain mask with T1 value
#         T2_GT_prime = torch.zeros_like(self.PD)
#         T2_GT_prime[self.mask_brain.squeeze(1)] = value_T2
#         PD_GT_prime = torch.zeros_like(self.PD)
#         PD_GT_prime[self.mask_brain.squeeze(1)] = value_PD
        
#         # calculate loss. between generated qmaps and GT_prime
#         self.loss_G_prime = 0
#         self.loss_G_prime += self.criterionL1(self.PD / 1, PD_GT_prime / 1)
#         self.loss_G_prime += self.criterionL1(self.T1 / 5, T1_GT_prime / 5)
#         self.loss_G_prime += self.criterionL1(self.T2 / 3, T2_GT_prime / 3)
#         # print(f'Loss_G_prime: {self.loss_G_prime.item()}')
#         self.loss_G_prime.backward()
        
#     def warmup_optimize_parameters(self):
#         self.forward()
#         self.optimizer_G.zero_grad()
#         self.warmup_backward_G()
#         self.optimizer_G.step()


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

        # Handle modality dropout / input selection
        scan_map = {'T1w': self.real_T1w, 'T2w': self.real_T2w, 'FLAIR': self.real_FLAIR}
        
        if 'shared_attn_unet' in self.opt.which_model_netG:
            # Modality dropout logic (simplified for brevity, assuming _choose_modalities logic exists or is inlined)
            # For this refactor, we assume the inputs are prepped correctly.
            self.G_inputs = self._prepare_inputs_with_dropout(scan_map)
        
        # Load Sequence parameters
        self.T1w_TR = input['T1w_TR'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.T1w_TE = input['T1w_TE'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        # self.T1w_TI = input['T1w_TI'].unsqueeze(-1).to(dtype=torch.float, device=self.device)
        self.T1w_TI = None
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
                    # Safety check: if user asked for T1w but dataset didn't provide it
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

        # 2. Extract Quantitative Maps (Q-Maps)
        # Apply scaling and masking immediately
        # self.Q1 = (self.fake_B[:, 0, :, :] * 1).unsqueeze(1) * self.mask_brain # PD
        # self.Q2 = (self.fake_B[:, 1, :, :] * 5).unsqueeze(1) * self.mask_brain # T1
        # self.Q3 = (self.fake_B[:, 2, :, :] * 3).unsqueeze(1) * self.mask_brain # T2

        # self.PD = self.Q1.squeeze(1)
        # self.T1 = self.Q2.squeeze(1)
        # self.T2 = self.Q3.squeeze(1)
        
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
        
        # DIAGNOSTIC CHECK
        if torch.isnan(self.fake_T1w_raw).any():
            print("NaN detected in Physics Synthesis!")
            print(f"Q1 min/max: {self.Q1.min().item()}/{self.Q1.max().item()}")
            print(f"Q2 min/max: {self.Q2.min().item()}/{self.Q2.max().item()}")
            print(f"Q3 min/max: {self.Q3.min().item()}/{self.Q3.max().item()}")
            
    def _batch_signal_model(
            self,
            PD, T1, T2,
            TR, TE, TI, FA,
            seq_types,
            TI2=None):
        """
        Apply `self.signal_model` one sample at a time so that every
        element can have its *own* `seq_type`.

        tensors: B×H×W (no channel dim, you already stripped it above)
        seq_types:  list  (length == B)  or  1-D tensor of dtype=object/str
        """
        outs = []
        B = PD.size(0)
        for b in range(B):
            out = self.signal_model(
                PD[b], T1[b], T2[b],
                TR[b], TE[b],
                None if TI  is None else TI[b],
                FA[b],
                seq_type=str(seq_types[b]),          
                TI2=None if TI2 is None else TI2[b],
            )
            out = out.squeeze(0)
            outs.append(out)
        return torch.stack(outs, dim=0)              # B×1×H×W

    def signal_model(self, PD, T1, T2, TR, TE, TI, FA, seq_type='TFE', TI2=None):
        TR = TR / 1000
        TE = TE / 1000
        if TI is not None:
            TI = TI / 1000
        if TI2 is not None:
            TI2 = TI2 / 1000
        TR = TR.unsqueeze(-1)
        TE = TE.unsqueeze(-1)
        if TI is not None:
            TI = TI.unsqueeze(-1)
        if TI2 is not None:
            TI2 = TI2.unsqueeze(-1)
        FA = FA.unsqueeze(-1)
        FA = torch.deg2rad(FA)
        
        if seq_type == 'MPRAGE':
            out = torch.abs(PD * (1 - 2 * torch.exp(-TI / (T1 + 1e-6)) + torch.exp(-TR / (T1 + 1e-6))) * torch.sin(FA) * torch.exp(-TE / (T2 + 1e-6)) / (1 + torch.cos(FA) * torch.exp(-TR / (T1 + 1e-6))))
        elif seq_type == 'TFE':
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

    # def _batch_signal_model(self, PD, T1, T2, TR, TE, TI, FA, seq_types):
    #     outs = []
    #     B = PD.size(0)
    #     for b in range(B):
    #         out = self.signal_model(PD[b], T1[b], T2[b], TR[b], TE[b], 
    #                             None if TI is None else TI[b], FA[b], 
    #                             seq_type=str(seq_types[b]))
            
    #         if torch.isnan(out).any():
    #             subject_id = self.subject_names[b] if hasattr(self, 'subject_names') else b
    #             print(f"ERROR: Physics Synthesis produced NaN for subject {subject_id}")
    #             # Optional: print min/max of Q-maps to see if they exploded
    #             print(f"T1 range: {T1[b].min().item():.3f} to {T1[b].max().item():.3f}")
    #             raise FloatingPointError(f"NaN in physics model for {subject_id}")
                
    #         outs.append(out.squeeze(0))
    #     return torch.stack(outs, dim=0).unsqueeze(1)

    # def signal_model(self, PD, T1, T2, TR, TE, TI, FA, seq_type='TFE'):
    #     TR, TE = TR/1000, TE/1000
    #     if TI is not None: TI = TI/1000
    #     FA = torch.deg2rad(FA.unsqueeze(-1))
    #     print(f"Signal model called for seq_type: {seq_type}, with shapes PD: {PD.shape}, T1: {T1.shape}, T2: {T2.shape}, TR: {TR.shape}, TE: {TE.shape}, TI: {TI.shape if TI is not None else 'N/A'}, FA: {FA.shape}")
    #     print(f"Sequence parameters sample - TR: {TR.flatten()[0].item()}, TE: {TE.flatten()[0].item()}, TI: {TI.flatten()[0].item() if TI is not None else 'N/A'}, FA: {FA.flatten()[0].item()} rad")
             
    #     if seq_type == 'MPRAGE':
    #         out = torch.abs(PD * (1 - 2 * torch.exp(-TI / (T1 + 1e-6)) + torch.exp(-TR / (T1 + 1e-6))) * torch.sin(FA) * torch.exp(-TE / (T2 + 1e-6)) / (1 + torch.cos(FA) * torch.exp(-TR / (T1 + 1e-6))))
    #     elif seq_type == 'TFE' or seq_type == 'GRE':
    #         out = PD * torch.sin(FA) * (1 - torch.exp(-TR / (T1+1e-6))) * torch.exp(-TE / (T2+1e-6)) / (1 - torch.cos(FA) * torch.exp(-TR / (T1+1e-6)))
    #     elif seq_type == 'SE':
    #         out =  PD * torch.sin(FA) * (1 - torch.exp(-TR / (T1+1e-6))) * torch.exp(-TE / (T2+1e-6)) / (1 - torch.cos(FA) * torch.exp(-TR / (T1+1e-6)))
    #     elif seq_type == 'TSE':
    #         out = PD * torch.exp(-TE / (T2 + 1e-6)) * (1 - torch.exp(-TR / (T1 + 1e-6)))
    #     elif seq_type == 'IR' or seq_type == 'FLAIR':
    #         out = torch.abs(PD * torch.exp(-TE / (T2 + 1e-6)) * (1 - 2 * torch.exp(-TI / (T1 + 1e-6)) + torch.exp(-TR / (T1 + 1e-6))))
    
    #     if out.dim() == 2: out = out.unsqueeze(0)
    #     return out

    def _apply_rescaling(self, fake, real, bounds_percent, exact_factor):
        """Calculates and applies rescaling factor. Returns rescaled fake and the factor."""
        method = self.opt.rescaling_method
        
        # Pre-calculate bounds (used only if method is bounded)
        bounds = (exact_factor - bounds_percent * exact_factor, exact_factor + bounds_percent * exact_factor)
        lower, upper = bounds[0].squeeze(1), bounds[1].squeeze(1)

        factor = None

        if method == 'mean':
            # Use GM+WM mask if available (restored logic), else whole brain
            mask = self.mask_gm_wm.float() 
            mean_fake = (fake * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)
            mean_real = (real * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)
            factor = mean_real / (mean_fake + 1e-6)
            # Apply bounds for 'mean' method as per original code convention
            factor = torch.clamp(factor, min=lower, max=upper)

        elif method == 'closed_form_least_squares':
            # --- RESTORED UNBOUNDED METHOD ---
            # Minimizes || Real - alpha * Fake ||^2
            # alpha = sum(Mask * Fake * Real) / sum(Mask * Fake^2)
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
        
    def _find_nan_culprit(self):
        """Iterates through the batch to find which subject caused the NaN."""
        for i in range(self.PD.size(0)):
            # Check the primary outputs of the network for this batch item
            nan_pd = torch.isnan(self.PD[i]).any()
            nan_t1 = torch.isnan(self.T1[i]).any()
            nan_t2 = torch.isnan(self.T2[i]).any()
            
            if nan_pd or nan_t1 or nan_t2:
                subject_id = self.subject_names[i] if hasattr(self, 'subject_names') else f"Batch Index {i}"
                print(f"CRITICAL: NaN found in Q-Maps for Subject: {subject_id}")
                print(f" -> PD has NaN: {nan_pd}")
                print(f" -> T1 has NaN: {nan_t1}")
                print(f" -> T2 has NaN: {nan_t2}")

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        loss_G, metrics = self.compute_losses_and_metrics(optimize=True)

        # --- NAN CHECK START ---
        if torch.isnan(loss_G):
            print("\n[!] NaN detected in Total Loss!")
            # 'input' is the batch dictionary passed to set_input
            # We assume you store a reference to the current batch in self.last_input 
            # or similar during set_input for debugging.
            self._find_nan_culprit()
            raise RuntimeError("Stopping execution due to NaN loss.")
        # --- NAN CHECK END ---

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