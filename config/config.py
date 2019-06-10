import os
from os import path
import fastai
from fastai import *

class BaseOptions():
	pass

DatasetPaired = BaseOptions()
DatasetPaired.dataroot = '/Projects/NIR_FR_PTH/Oulu_ALIGN'
DatasetPaired.dataset_name = 'oulu'
DatasetPaired.max_dataset_size = 100000
DatasetPaired.preprocess = 'resize_and_crop' # 'resize_and_crop' or 'scale_width_and_crop'
DatasetPaired.load_size = 112
DatasetPaired.crop_size = 112
DatasetPaired.no_flip = False


DatasetUnpaired = BaseOptions()
DatasetUnpaired.dataroot = '/Projects/NIR_FR_PTH/CASIA_ALIGN'
DatasetUnpaired.dataset_name = 'casia'
DatasetUnpaired.max_dataset_size = 100000
DatasetUnpaired.preprocess = 'resize_and_crop' # 'resize_and_crop' or 'scale_width_and_crop'
DatasetUnpaired.load_size = 112
DatasetUnpaired.crop_size = 112
DatasetUnpaired.no_flip = False
DatasetUnpaired.serial_batches = False # for unaligned dataset only


ModelGan = BaseOptions()
ModelGan.pretrainModel = '/Projects/mk_utils/Convert_Mxnet_to_Pytorch/Pytorch_NewModel.pth'
ModelGan.ftExtractorCutNum = -3
ModelGan.norm_type_gen = fastai.vision.NormType.Spectral
ModelGan.bn_eps = 2e-5
ModelGan.bn_mom = 0.9
ModelGan.nf_factor = 2
ModelGan.wd = 1e-3
ModelGan.blur = True
ModelGan.blur_final = True
ModelGan.self_attention = True
ModelGan.last_cross = True
ModelGan.bottle = False
ModelGan.y_range = (-3.,3.)
ModelGan.num_classes_gen = 3
ModelGan.hook_detach = False
ModelGan.checkPointPath = None
ModelGan.checkPointName = 'chkpt'
ModelGan.checkPointNum = 0


ModelFtExtractor = BaseOptions()
ModelFtExtractor.pretrainModel = '/Projects/mk_utils/Convert_Mxnet_to_Pytorch/Pytorch_NewModel.pth'
ModelFtExtractor.ftDim = 512
ModelGan.checkPointPath = None
ModelGan.checkPointName = 'chkpt'
ModelGan.checkPointNum = 0