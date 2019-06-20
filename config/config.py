import os
from os import path
import fastai
from fastai import *
import torch
from torch import nn
import functools

def _resnet20_split(m:nn.Module): return (m[0][10],m[1])

class BaseOptions():
	pass


'''
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        

        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
'''

isTrain = True
continue_train = False # load model and/or optimizer from checkpoints if set to True
load_iter = 1
epoch = 1 # total epoch count
epoch_count = 1 # starting epoch count
niter = 1 # num of iter at starting learning rate
niter_decay = 0 # num of iter to linearly decay learning rate to zero
gpu_ids = [1]
train_bn = True
#freeze_gan_encoder = True
checkpoints_dir = '/Projects/NIR_FR_PTH/output'
name = 'train1'
verbose = False # if specified, print more debugging information
suffix = '' #customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}
save_epoch_freq = 1
batch_size = 16
serial_batches = False # set to False to shuffle the dataset
num_threads = 4 # num of threads for loading the data
preprocess = 'resize_and_crop' # 'resize_and_crop' or 'scale_width_and_crop'
dataset_name1 = 'PairedLabelDataset'
dataset_name2 = 'UnpairedLabelDataset'
data_lib = 'mklib.nn.pthnet.pthdataset' # dataset library file
discrim_input_size = 112
feature_loss_name = 'resnet20' # 'vgg16', 'resnet20'
if feature_loss_name == 'resnet20':
	resnet20_model_path = '/Projects/mk_utils/Convert_Mxnet_to_Pytorch/Pytorch_NewModel.pth'

# visdom and HTML visualization parameters
display_freq = 400 #frequency of showing training results on screen
display_ncols = 4 # if positive, display all images in a single visdom web panel with certain number of images per row.
display_id = 1 # window id of the web display
display_server = "http://localhost" # visdom server of the web display
display_env = 'main' # visdom display environment name (default is "main")
display_port = 8097 # visdom port of the web display
display_winsize = 256 # display window size for both visdom and HTML
update_html_freq = 1000 # frequency of saving training results to html
print_freq = 100 # frequency of showing training results on console
no_html = False # do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/

DatasetPaired = BaseOptions()
DatasetPaired.dataroot = '/Projects/NIR_FR_PTH/data/Oulu_ALIGN'
DatasetPaired.generatedroot = None # for imaged generated from GAN, set to None if don't want to include it
DatasetPaired.folderlevel = 2 # number of subfolder level under generatedroot
DatasetPaired.dataset_name = 'oulu'
DatasetPaired.max_dataset_size = 100000
DatasetPaired.preprocess = preprocess
DatasetPaired.load_size = 112
DatasetPaired.crop_size = 112
DatasetPaired.no_flip = False
if feature_loss_name == 'vgg16':
	DatasetPaired.mean = [0.485, 0.456, 0.406] # in r,g,b order
	DatasetPaired.std = [0.229, 0.224, 0.225] # in r,g,b order
elif feature_loss_name == 'resnet20':
	DatasetPaired.mean = [0.5, 0.5, 0.5] # in r,g,b order
	DatasetPaired.std = [0.5, 0.5, 0.5] # in r,g,b order
else:
	raise NotImplementedError("")
#DatasetPaired.mean_r = 138.0485
#DatasetPaired.mean_g = 110.2243
#DatasetPaired.mean_b = 96.73112

DatasetPairedGan = BaseOptions()
DatasetPairedGan.dataroot = '/Projects/NIR_FR_PTH/data/Oulu_ALIGN'
DatasetPairedGan.generatedroot = '/Projects/NIR_FR_PTH/data/Oulu_ALIGN/NI_GAN/Strong' # for imaged generated from GAN, set to None if don't want to include it
DatasetPairedGan.folderlevel = 2 # number of subfolder level under generatedroot
DatasetPairedGan.dataset_name = 'oulu'
DatasetPairedGan.max_dataset_size = 100000
DatasetPairedGan.preprocess = preprocess
DatasetPairedGan.load_size = 112
DatasetPairedGan.crop_size = 112
DatasetPairedGan.no_flip = False
if feature_loss_name == 'vgg16':
	DatasetPairedGan.mean = [0.485, 0.456, 0.406] # in r,g,b order
	DatasetPairedGan.std = [0.229, 0.224, 0.225] # in r,g,b order
elif feature_loss_name == 'resnet20':
	DatasetPairedGan.mean = [0.5, 0.5, 0.5] # in r,g,b order
	DatasetPairedGan.std = [0.5, 0.5, 0.5] # in r,g,b order
else:
	raise NotImplementedError("")

DatasetUnpaired = BaseOptions()
DatasetUnpaired.dataroot = '/Projects/NIR_FR_PTH/data/CASIA_ALIGN'
DatasetUnpaired.generatedroot = None # for imaged generated from GAN, set to None if don't want to include it
DatasetUnpaired.folderlevel = 0 # number of subfolder level under generatedroot
DatasetUnpaired.dataset_name = 'casia'
DatasetUnpaired.max_dataset_size = 100000
DatasetUnpaired.preprocess = preprocess
DatasetUnpaired.load_size = 112
DatasetUnpaired.crop_size = 112
DatasetUnpaired.no_flip = False
DatasetUnpaired.serial_batches = False # for unaligned dataset only
if feature_loss_name == 'vgg16':
	DatasetUnpaired.mean = [0.485, 0.456, 0.406] # in r,g,b order
	DatasetUnpaired.std = [0.229, 0.224, 0.225] # in r,g,b order
elif feature_loss_name == 'resnet20':
	DatasetUnpaired.mean = [0.5, 0.5, 0.5] # in r,g,b order
	DatasetUnpaired.std = [0.5, 0.5, 0.5] # in r,g,b order
else:
	raise NotImplementedError("")
#DatasetUnpaired.mean_r = 138.0485
#DatasetUnpaired.mean_g = 110.2243
#DatasetUnpaired.mean_b = 96.73112

DatasetUnpairedGan = BaseOptions()
DatasetUnpairedGan.dataroot = '/Projects/NIR_FR_PTH/data/CASIA_ALIGN'
DatasetUnpairedGan.generatedroot = '/Projects/NIR_FR_PTH/data/CASIA_ALIGN/NIR_GAN' # for imaged generated from GAN, set to None if don't want to include it
DatasetUnpairedGan.folderlevel = 0 # number of subfolder level under generatedroot
DatasetUnpairedGan.dataset_name = 'casia'
DatasetUnpairedGan.max_dataset_size = 100000
DatasetUnpairedGan.preprocess = preprocess
DatasetUnpairedGan.load_size = 112
DatasetUnpairedGan.crop_size = 112
DatasetUnpairedGan.no_flip = False
DatasetUnpairedGan.serial_batches = False # for unaligned dataset only
if feature_loss_name == 'vgg16':
	DatasetUnpairedGan.mean = [0.485, 0.456, 0.406] # in r,g,b order
	DatasetUnpairedGan.std = [0.229, 0.224, 0.225] # in r,g,b order
elif feature_loss_name == 'resnet20':
	DatasetUnpairedGan.mean = [0.5, 0.5, 0.5] # in r,g,b order
	DatasetUnpairedGan.std = [0.5, 0.5, 0.5] # in r,g,b order
else:
	raise NotImplementedError("")

DatasetMixed = BaseOptions()
DatasetMixed.dataroot = ['/Projects/NIR_FR_PTH/data/CASIA_ALIGN/NIR', '/Projects/NIR_FR_PTH/data/CASIA_ALIGN/VIS']
DatasetMixed.label = [0, 1] # 0 is for generated, 1 is for real
assert len(DatasetMixed.dataroot) == len(DatasetMixed.label)
DatasetMixed.max_dataset_size = 100000
DatasetMixed.preprocess = preprocess
DatasetMixed.load_size = 112
DatasetMixed.crop_size = 112
DatasetMixed.no_flip = False
DatasetMixed.serial_batches = False # for unaligned dataset only
if feature_loss_name == 'vgg16':
	DatasetMixed.mean = [0.485, 0.456, 0.406] # in r,g,b order
	DatasetMixed.std = [0.229, 0.224, 0.225] # in r,g,b order
elif feature_loss_name == 'resnet20':
	DatasetMixed.mean = [0.5, 0.5, 0.5] # in r,g,b order
	DatasetMixed.std = [0.5, 0.5, 0.5] # in r,g,b order
else:
	raise NotImplementedError("")

ModelGan = BaseOptions()
ModelGan.pretrainModel = '/Projects/mk_utils/Convert_Mxnet_to_Pytorch/Pytorch_NewModel.pth'
ModelGan.ftExtractorCutNum = -3
ModelGan.norm_type_gen = fastai.vision.NormType.Spectral
ModelGan.bn_eps = 2e-5
ModelGan.bn_mom = 0.1
ModelGan.nf_factor = 2
ModelGan.blur = True
ModelGan.blur_final = True
ModelGan.self_attention = True
ModelGan.last_cross = True
ModelGan.bottle = False
ModelGan.num_classes_gen = 3
ModelGan.hook_detach = False # default is False
ModelGan.checkPointPath = None
ModelGan.checkPointName = 'chkpt'
ModelGan.checkPointNum = 0
ModelGan.split_on = _resnet20_split
ModelGan.opt_func = functools.partial(torch.optim.Adam, betas=(0.9,0.999))
ModelGan.wd = 1e-2
ModelGan.true_wd = True
ModelGan.bn_wd = True
ModelGan.lr = slice(1e-3)
ModelGan.lr_decay_iters = 50 # multiply by a gamma every lr_decay_iters iterations
ModelGan.lr_policy = 'linear'
if feature_loss_name == 'vgg16':
	ModelGan.y_range = (-3.,3.)
elif feature_loss_name == 'resnet20':
	ModelGan.y_range = (-1.,1.)
else:
	raise NotImplementedError("")


ModelCritic = BaseOptions()
ModelCritic.pretrainModel = '/Projects/mk_utils/Convert_Mxnet_to_Pytorch/Pytorch_NewModel.pth'
ModelCritic.ftExtractorCutNum = -3
ModelCritic.bn_eps = 2e-5
ModelCritic.bn_mom = 0.1
ModelCritic.discriminator_type = 'pair' # 'pair', 'unpair', 'both'
ModelCritic.checkPointPath = None
ModelCritic.checkPointName = 'chkpt'
ModelCritic.checkPointNum = 0
ModelCritic.opt_func = functools.partial(torch.optim.Adam, betas=(0.9,0.999))
ModelCritic.wd = 1e-2
ModelCritic.true_wd = True
ModelCritic.bn_wd = True
ModelCritic.lr = 1e-3 # do not use slice here