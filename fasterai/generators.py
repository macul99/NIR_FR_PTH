from fastai.vision import *
from fastai.vision.learner import cnn_config
from .unet import DynamicUnetWide, DynamicUnetDeep
from .loss import FeatureLoss
from .dataset import *
#from mklib.nn.pthnet import pthutils, pthunet

#Weights are implicitly read from ./models/ folder 
def gen_inference_wide(root_folder:Path, weights_name:str, nf_factor:int=2, arch=models.resnet101)->Learner:
      data = get_dummy_databunch()
      learn = gen_learner_wide(data=data, gen_loss=F.l1_loss, nf_factor=nf_factor, arch=arch)
      learn.path = root_folder
      learn.load(weights_name)
      learn.model.eval()
      return learn

def gen_learner_wide(data:ImageDataBunch, gen_loss=FeatureLoss(), arch=models.resnet101, nf_factor:int=2)->Learner:
    return unet_learner_wide(data, arch=arch, wd=1e-3, blur=True, norm_type=NormType.Spectral,
                        self_attention=True, y_range=(-3.,3.), loss_func=gen_loss, nf_factor=nf_factor)

#The code below is meant to be merged into fastaiv1 ideally
def unet_learner_wide(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, 
                 blur:bool=False, self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True,
                 bottle:bool=False, nf_factor:int=1, **kwargs:Any)->Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)
    #preModel = torch.load('/Projects/mk_utils/Convert_Mxnet_to_Pytorch/Pytorch_NewModel.pth')
    #preModel = list(preModel.children())[0]
    #body = pthutils.cut_model(preModel,-3)

    model = to_device(DynamicUnetWide(body, n_classes=data.c, blur=blur, blur_final=blur_final,
          self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
          bottle=bottle, nf_factor=nf_factor), data.device)
    #print(model[2])
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on,meta['split']))
    if pretrained: learn.freeze_to(-2)
    #print(model[2])
    #apply_init(model[2], nn.init.kaiming_normal_)
    return learn

#----------------------------------------------------------------------

#Weights are implicitly read from ./models/ folder 
def gen_inference_deep(root_folder:Path, weights_name:str, arch=models.resnet34, nf_factor:float=1.5)->Learner:
      data = get_dummy_databunch()
      learn = gen_learner_deep(data=data, gen_loss=F.l1_loss, arch=arch, nf_factor=nf_factor)
      learn.path = root_folder
      learn.load(weights_name)
      learn.model.eval()
      return learn

def gen_learner_deep(data:ImageDataBunch, gen_loss=FeatureLoss(), arch=models.resnet34, nf_factor:float=1.5)->Learner:
    return unet_learner_deep(data, arch, wd=1e-3, blur=True, norm_type=NormType.Spectral,
                        self_attention=True, y_range=(-3.,3.), loss_func=gen_loss, nf_factor=nf_factor)

#The code below is meant to be merged into fastaiv1 ideally
def unet_learner_deep(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, 
                 blur:bool=False, self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True,
                 bottle:bool=False, nf_factor:float=1.5, **kwargs:Any)->Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)
    model = to_device(DynamicUnetDeep(body, n_classes=data.c, blur=blur, blur_final=blur_final,
          self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
          bottle=bottle, nf_factor=nf_factor), data.device)
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on,meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn

#-----------------------------