# encoding: utf-8
import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict
import jittor as jt

from .logger import get_logger

jt.flags.use_cuda = 1

logger = get_logger()

# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


# def reduce_tensor(tensor, dst=0, op=dist.ReduceOp.SUM, world_size=1):
#     tensor = tensor.clone()
#     dist.reduce(tensor, dst, op)
#     if dist.get_rank() == dst:
#         tensor.div_(world_size)

#     return tensor


# def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
#     tensor = tensor.clone()
#     dist.all_reduce(tensor, op)
#     if norm:
#         tensor.div_(world_size)

#     return tensor


def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        state_dict = jt.load(model_file)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model



def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system('ln -s {} {}'.format(src, target))


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _dbg_interactive(var, value):
    from IPython import embed
    embed()
