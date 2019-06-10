import fastai
from fastai import *
from fastai.core import *
from fastai.vision.transform import get_transforms
from fastai.vision.data import ImageImageList, ImageDataBunch, imagenet_stats
from .augs import noisify


def get_colorize_data(sz:int, bs:int, crappy_path:Path, good_path:Path, random_seed:int=None, 
        keep_pct:float=1.0, num_workers:int=8, xtra_tfms=[])->ImageDataBunch:

    src = (ImageImageList.from_folder(crappy_path)
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .split_by_rand_pct(0.1, seed=random_seed))
    print('src_type: ', type(src))
    data = (src.label_from_func(lambda x: good_path/x.relative_to(crappy_path)))
    print('data.train.x type: ', type(data.train.x))
    data = (data.transform(get_transforms(max_zoom=1.2, max_lighting=0.5, max_warp=0.25, xtra_tfms=xtra_tfms), size=sz, tfm_y=True))
    print('data.train.x type: ', type(data.train.x))
    data = (data.databunch(bs=bs, num_workers=num_workers, no_check=True)
        .normalize(imagenet_stats, do_y=True))
    print('data_type3: ', type(data))
    data.c = 3
    return data



def get_dummy_databunch()->ImageDataBunch:
    path = Path('./dummy/')
    return get_colorize_data(sz=1, bs=1, crappy_path=path, good_path=path, keep_pct=0.001)
