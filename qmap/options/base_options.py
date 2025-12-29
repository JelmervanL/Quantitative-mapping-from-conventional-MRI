import argparse
import os
from qmap.util import util
import qmap.models as models
import qmap.data as data
from qmap.util.util import _parse_int_list 
import torch
import yaml

class BaseOptions:
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        # --- config argument ---
        parser.add_argument('--config', type=str, default='configs/train_umcu_paper1.yaml', help='Path to YAML config file')
        
        # --- Checkpointing ---
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_latest_freq', type=int, default=5, help='frequency of saving the latest model')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix')

        # --- Data IO ---
        parser.add_argument('--dataroot', type=lambda s: s.split(','), required=False, help="One or more dataset roots")
        parser.add_argument('--data_csv_name',type=lambda s: [p.strip() for p in s.split(',')], required=False, help="CSV filenames")        
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--max_queuelength', type=int, default=100, help='max queue length torchio')
        parser.add_argument('--patches_per_volume', type=int, default=5, help='number of patches per volume')
        parser.add_argument('--padcrop', type=int, default=224, help='pad or crop images to this pixel size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--input_scan_types', type=lambda s: [x.strip() for x in s.split(',')], default=['T1w', 'T2w', 'FLAIR'], help='Scan types')
        parser.add_argument('--use_volunteer_dataset', action='store_true', help='use volunteer dataset for testing')
        parser.add_argument('--train_subset_size', type=int, default=0,help='randomly select this many subjects')
        parser.add_argument('--eval_size', type=int, default=10000000, help='number of evaluation slices')

        # --- Processing / Q-Star ---
        parser.add_argument('--rescaling_method', type=str, default='mean', help='rescaling method')
        parser.add_argument('--rescaling_factor_bounds', type=float, default=0.2, help='rescaling factor bounds')
        parser.add_argument('--eval_qstar_tissue_values', action='store_true', help='evaluate qstar tissue values')
        
        # --- Model Architecture ---
        parser.add_argument('--model', type=str, default='qmap_synth', help='chooses which model to use')
        parser.add_argument('--which_model_netG', type=str, default='res_unet', help='selects model to use for netG')
        parser.add_argument('--feature_channels_G',  type=_parse_int_list, default=[64, 128, 256], help='encoder stage widths')
        parser.add_argument('--norm_G', type=str, default='instance', help='normalization for G')
        parser.add_argument('--encoder_norm', type=str, default='instance', choices=['instance', 'batch', 'group', 'none'])
        parser.add_argument('--bottleneck_norm', type=str, default='instance', choices=['instance', 'batch', 'group', 'none'])
        parser.add_argument('--decoder_norm', type=str, default='instance', choices=['instance', 'batch', 'group', 'none'])
        parser.add_argument('--dropout_prob_decoder', type=float, default=0.0, help='Dropout prob decoder')
        parser.add_argument('--dropout_prob_bottleneck', type=float, default=0.0, help='Dropout prob bottleneck')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for init')

        # --- Training Loop ---
        parser.add_argument('--phase', type=str, default='train', help='train, val, test')
        parser.add_argument('--do_val', action='store_true', help='do validation')
        parser.add_argument('--n_epochs', type=int, default=100, help='# of training epochs')
        parser.add_argument('--epoch_count_start', type=int, default=1, help='the starting epoch count')
        parser.add_argument('--which_epoch_load', type=str, default='latest', help='which epoch to load')
        parser.add_argument('--n_warmup_epochs', type=int, default=0, help='number of warm up epochs')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
        parser.add_argument('--random_seed', type=int, default=0, help='random seed')
        parser.add_argument('--single_subject', action='store_true', help='debug single subject')
        parser.add_argument('--continue_train', action='store_true', help='continue training')

        # --- Optimization ---
        parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'])
        parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
        parser.add_argument('--lr_policy', type=str, default='none', help='learning rate policy')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay lr')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--use_mixed_precision', action='store_true', help='use mixed precision')

        # --- Losses ---
        parser.add_argument('--loss_content_I_l1', type=float, default=0, help='content loss, l1')
        parser.add_argument('--loss_content_I_l2', type=float, default=0, help='content loss, l2')
        parser.add_argument('--loss_content_pearson', type=float, default=0, help='content loss, pearson')
        parser.add_argument('--loss_vgg', type=float, default=0.0, help='weight of perceptual loss')
        parser.add_argument('--loss_PD_constraint', type=float, default=0, help='constraint of PD')
        parser.add_argument('--loss_PD_prior', type=float, default=0, help='prior of PD (for mask loading)')
        parser.add_argument('--loss_tv_reg', type=float, default=0, help='total variation regularization')
        
        # --- Modality Dropout ---
        parser.add_argument('--modality_dropout_rate', type=float, default=0.0, help='Dropout rate')
        parser.add_argument('--modality_dropout_max_missing', type=int, default=2, help='Max missing')
        parser.add_argument('--modality_dropout_burnin', type=int, default=0, help='Burn-in epochs')
        parser.add_argument("--oversample_T1w_SE", action="store_true", default=False, help="Oversample minority")

        # --- Logging ---
        parser.add_argument('--verbose', action='store_true', help='print debugging info')
        parser.add_argument('--use_wandb', action='store_true', help='use wandb')
        parser.add_argument('--wandb_project', type=str, default='qmap', help='wandb project name')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of console print')
        parser.add_argument('--save_images_visualizer', action='store_true', help='save images')
        parser.add_argument('--save_val_nifti', action='store_true', help='save nifti files')
        parser.add_argument('--calc_additional_metrics', action='store_true', help='calc PSNR/SSIM')

        self.initialized = True
        return parser

    def gather_options(self):
        import yaml  # Ensure yaml is imported

        # 1. Initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # 2. Check for the --config argument first
        args, _ = parser.parse_known_args()
        
        config_args = {}
        if args.config:
            print(f"Loading configuration from {args.config}")
            with open(args.config, 'r') as f:
                config_args = yaml.safe_load(f)
            parser.set_defaults(**config_args)

        if 'model' in config_args:
             parser.set_defaults(model=config_args['model'])

        # Parse again to get the correct model name (CLI > Config > Default)
        opt_temp, _ = parser.parse_known_args()
        model_name = opt_temp.model
        
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser)
        
        # 4. Final Parse
        self.parser = parser
        opt = parser.parse_args()

        # 5. Manual Post-Processing for List Arguments
        if isinstance(opt.dataroot, str):
            opt.dataroot = opt.dataroot.split(',')
        
        if isinstance(opt.data_csv_name, str):
            opt.data_csv_name = [x.strip() for x in opt.data_csv_name.split(',')]
            
        if isinstance(opt.input_scan_types, str):
            opt.input_scan_types = [x.strip() for x in opt.input_scan_types.split(',')]
            
        if isinstance(opt.feature_channels_G, str):
            opt.feature_channels_G = [int(x) for x in opt.feature_channels_G.split(',') if x.strip()]
        
        self.opt = opt
        return self.opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if len(opt.dataroot) != len(opt.data_csv_name):
            raise ValueError(
                f"--dataroot has {len(opt.dataroot)} entries, but --data_csv_name has "
                f"{len(opt.data_csv_name)}; they must match."
            )

        # 1. If it's already a list (e.g., [0] from YAML or previous processing), use it
        if isinstance(opt.gpu_ids, list):
             opt.gpu_ids = [int(x) for x in opt.gpu_ids]
        # 2. If it's a single integer (e.g., 0 from YAML), wrap it
        elif isinstance(opt.gpu_ids, int):
             opt.gpu_ids = [opt.gpu_ids]
        # 3. If it's a string (e.g., "0" from CLI), cast and wrap
        else:
             opt.gpu_ids = [int(opt.gpu_ids)]
            
        # Set the device if ID is valid (>= 0)
        if len(opt.gpu_ids) > 0 and opt.gpu_ids[0] >= 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        else:
            opt.gpu_ids = []
        # -------------------------

        self.opt = opt
        return self.opt
