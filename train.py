import argparse

import jittor as jt
import jittor.nn as nn
from jittor import optim
import numpy as np
import sys
import os
from tqdm import tqdm
import os.path as osp
import networks
from dataset.datasets import CSDataSet,ADEDataSet

from tensorboardX import SummaryWriter
# from utils.pyt_utils import decode_labels, inv_preprocess, decode_predictions
from loss.criterion import CriterionDSN, CriterionOhemDSN
from engine import Engine

jt.flags.use_cuda = 1

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
ADE_IMG_MEAN = np.array((103.5, 116.28, 123.675), dtype=np.float32)
ADE_IMG_STD = np.array((0.225, 0.224, 0.229), dtype=np.float32)
BATCH_SIZE = 8
DATA_DIRECTORY = 'cityscapes'
DATA_LIST_PATH = './dataset/list/cityscapes/train.lst'
IGNORE_LABEL = 255
INPUT_SIZE = '769,769'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 40000
POWER = 0.9
RANDOM_SEED = 12345
RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_init.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 30000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
IMG_MAX_SIZE = 600

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--print-frequency", type=int, default=50,
                        help="Number of training steps.") 
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="choose the number of workers.")
    parser.add_argument("--recurrence", type=int, default=0,
                        help="choose the number of recurrence.")

    parser.add_argument("--ohem", type=str2bool, default='False',
                        help="use hard negative mining")
    parser.add_argument("--ohem-thres", type=float, default=0.6,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--ohem-keep", type=int, default=200000,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--datasets", type=str, default='cityscapes',
                    help="select the dataset to use. cityscapes and ade is available now.")
    parser.add_argument("--img_max_size", type=int, default=IMG_MAX_SIZE,
                        help="choose the max size of images in ADE20k")
    parser.add_argument("--need_crop", action="store_true",
                    help="Whether to crop the image in ADE20K.")
    parser.add_argument("--batch_per_gpu", type=int, default=1, 
                    help="if you want to run 8 batch in 4 gpu, batch_per_gpu is 2")
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
            
def adjust_learning_rate(optimizer, learning_rate, i_iter, max_iter, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003

def main():
    """Create the model and start the training."""
    parser = get_parser()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        
        # todo
        # cudnn.benchmark = True
        seed = args.random_seed
        # if engine.distributed:
        #     seed = engine.local_rank
        jt.set_global_seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(seed)

        # data loader
        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)
        if args.datasets == 'cityscapes':
            dataset = CSDataSet(args.data_dir, args.data_list, max_iters=None, crop_size=input_size, 
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
        elif args.datasets == 'ade':
            dataset = ADEDataSet(args.data_dir, args.data_list, max_iters=None, crop_size=input_size, 
            mirror=args.random_mirror, mean=IMG_MEAN, std=ADE_IMG_STD, is_train=True, need_crop=args.need_crop,
            max_img_size=args.img_max_size)
        train_loader = engine.get_train_loader(dataset)

        # config network and criterion
        if args.ohem:
            criterion = CriterionOhemDSN(thresh=args.ohem_thres, min_kept=args.ohem_keep)
        else:
            criterion = CriterionDSN() #CriterionCrossEntropy()

        # model = Res_Deeplab(args.num_classes, criterion=criterion,
        #         pretrained_model=args.restore_from)
        seg_model = eval('networks.' + args.model + '.Seg_Model')(
            num_classes=args.num_classes, criterion=criterion,
            pretrained_model=args.restore_from, recurrence=args.recurrence
        )
        # seg_model.init_weights()

        # group weight and config optimizer
        optimizer = optim.SGD([{'params': seg_model.parameters(), 'lr': args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.zero_grad()

        # model = torch.nn.DataParallel(seg_model)#engine.data_parallel(seg_model) 
        model = seg_model
        model.train()

        if jt.rank == 0 and not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)
            
        run = True
        global_iteration = args.start_iters
        avgloss = 0

        while run:
            epoch = global_iteration // (len(train_loader)//jt.world_size)

            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(len(train_loader)//jt.world_size), file=sys.stdout,
                        bar_format=bar_format)
            dataloader = iter(train_loader)
            print('start')

            for idx in pbar:
                global_iteration += 1

                images, labels, _, _ = next(dataloader)
                labels = labels.long()
                
                lr = adjust_learning_rate(optimizer, args.learning_rate, global_iteration-1, args.num_steps, args.power)
                print('lr',lr)
                loss = model(images, labels).mean()

                reduce_loss = loss.data#engine.all_reduce_tensor(loss)
                optimizer.backward(loss / args.batch_per_gpu)
                if (idx+1) % args.batch_per_gpu == 0:
                    optimizer.step()
                    # optimizer.zero_grad()


                print_str = 'Epoch{}/Iters{}'.format(epoch, global_iteration) \
                        + ' Iter{}/{}:'.format(idx + 1, len(train_loader)//jt.world_size) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % reduce_loss.item()

                pbar.set_description(print_str, refresh=False)

                if (not jt.in_mpi) or (jt.in_mpi and jt.rank == 0):
                    if global_iteration % args.save_pred_every == 0 or global_iteration >= args.num_steps:
                        print('taking snapshot ...')
                        if args.datasets == 'cityscapes':
                            save_path = osp.join(args.snapshot_dir, 'CS_scenes_'+str(global_iteration) + '_' + args.model +'.pkl')
                        elif args.datasets == 'ade':
                            save_path = osp.join(args.snapshot_dir, 'ADE20k_'+str(global_iteration) + '_' + args.model + '.pkl')
                        jt.save(seg_model.state_dict(), save_path) 

                if global_iteration >= args.num_steps:
                    run = False
                    break    



if __name__ == '__main__':
    main()
