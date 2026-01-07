import os
import torch
from collections import OrderedDict
from . import networks


class BaseModel:
    @staticmethod
    def modify_commandline_options(parser):
        return parser
    
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        print('save_dir:', self.save_dir)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    def forward(self, is_train=True):
        pass

    def setup(self, opt):
        """ Load and print networks and create schedulers """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch_load)
        if opt.phase != 'val':
            self.print_networks(opt.verbose)

    def eval(self):
        """ Make models eval mode during test time """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """ Used in test time, wrapping 'forward' in no_grad() so we don't save intermediate steps for backprop """
        with torch.no_grad():
            self.forward(is_train=False)  # saves some time to only calculate metrics during validation
            self.get_val_losses()

    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        return lr

    def get_current_visuals(self):
        """ Return visualization images. train.py will display these images and save the images to a html """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)       
        return visual_ret

    def get_current_losses(self):
        """ Return traning losses/errors. train.py will print out these errors as debugging information """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):

                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, which_epoch):
        """ Save model to disk """
        for name in self.model_names:
            print( "Saving model:", name )
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                print( "Save path:", save_path )
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, which_epoch):
        """Load model from disk"""
        for name in self.model_names:
            if not isinstance(name, str):
                continue

            load_filename = f"{which_epoch}_net_{name}.pth"
            load_path = os.path.join(self.save_dir, load_filename)
            
            print(f"Attempting to load model checkpoint from: {load_path}")

            if not os.path.exists(load_path):
                print(f"Model checkpoint doesn’t exist: {load_path}")
                continue

            # Grab the network attribute; if it was DataParallel, unwrap to .module
            net = getattr(self, f"net{name}")
            if isinstance(net, torch.nn.DataParallel):
                net = net.module

            if self.opt.phase != "val":
                print(f"Loading the model from {load_path}")

            state_dict = torch.load(load_path, map_location=self.device)

            net.load_state_dict(state_dict, strict=True)
            
            print(f"Successfully loaded model weights for net{name}.")

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
                       
