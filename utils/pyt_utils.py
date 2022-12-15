# encoding: utf-8
import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict, abc
import jittor as jt
from jittor import nn

from itertools import repeat

from .logger import get_logger

# jt.flags.use_cuda = 1

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
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
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

def to_2tuple(x):
    if isinstance(x, abc.Iterable):
        return x
    return tuple(repeat(x, 2))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
    random_tensor.floor()  # binarize
    output = x / keep_prob * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.is_training())


def cross_entropy_loss(output, target, weight=None, ignore_index=None,reduction='mean'):
    target_shape = target.shape
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))

    target = target.reshape((-1, ))
    target_weight = jt.ones(target.shape[0], dtype='float32')
    if weight is not None:
        target_weight = weight[target]
    if ignore_index is not None:
        target_weight = jt.ternary(
            target==ignore_index,
            jt.array(0).broadcast(target_weight),
            target_weight
        )
    
    target = target.broadcast(output, [1])
    target = target.index(1) == target
    
    output = output - output.max([1], keepdims=True)
    logsum = output.exp().sum(1).log()
    loss = (logsum - (output*target).sum(1)) * target_weight

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        if target_weight.mean() > 0:
            return loss.mean() / target_weight.mean()
        else:
            return loss.mean()
    else:
        return loss.reshape(target_shape) 

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None):
        self.weight = weight
        self.ignore_index = ignore_index
        
    def execute(self, output, target):
        return cross_entropy_loss(output, target, self.weight, self.ignore_index)
