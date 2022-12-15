import os
import os.path as osp
import time
import argparse

import jittor as jt

from utils.logger import get_logger
from utils.pyt_utils import extant_file

from dataset.datasets import CSDataSet

logger = get_logger()

# jt.flags.use_cuda = 1

class Engine(object):
    def __init__(self, custom_parser=None):
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.continue_state_object = self.args.continue_fpath

        # if not self.args.gpu == 'None':
        #     os.environ["CUDA_VISIBLE_DEVICES"]=self.args.gpu

        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1

        if self.distributed:
            raise RuntimeError("Unimplemented for distributed")
            # self.local_rank = self.args.local_rank
            # self.world_size = int(os.environ['WORLD_SIZE'])
            # torch.cuda.set_device(self.local_rank)
            # dist.init_process_group(backend="nccl", init_method='env://')
            # self.devices = [i for i in range(self.world_size)]
        else:
            gpus = os.environ["CUDA_VISIBLE_DEVICES"]
            self.devices =  [i for i in range(len(gpus.split(',')))] 

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=extant_file,
                       metavar="FILE",
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        p.add_argument('--local_rank', default=0, type=int,
                       help='process rank on node')

    def get_train_loader(self, train_dataset):
        is_shuffle = True
        batch_size = self.args.batch_size

        train_dataset.set_attrs(batch_size=batch_size,
                                num_workers=self.args.num_workers,
                                drop_last=True,
                                shuffle=is_shuffle)
                                #    pin_memory=True)
        train_loader = train_dataset
        return train_loader

    def get_test_loader(self, test_dataset):
        is_shuffle = False
        batch_size = self.args.batch_size

        test_dataset.set_attrs(batch_size=batch_size,
                            num_workers=self.args.num_workers,
                            drop_last=True,
                            shuffle=is_shuffle)
        test_loader = test_dataset              
        return test_loader


    def all_reduce_tensor(self, tensor, norm=True):
        return jt.mean(tensor)


    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        jt.clean_graph()
        jt.sync_all()
        jt.gc()
        jt.display_memory_info()
        if type is not None:
            logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False
