import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
from tqdm import tqdm

import jittor as jt
import jittor.nn as nn
from dataset.datasets import CSDataSet, ADEDataSet
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage
from utils.pyt_utils import load_model
import networks
from engine import Engine

jt.flags.use_cuda = 1

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
ADE_IMG_MEAN = np.array((103.5, 116.28, 123.675), dtype=np.float32)
ADE_IMG_STD = np.array((0.225, 0.224, 0.229), dtype=np.float32)
DATA_DIRECTORY = 'cityscapes'
DATA_LIST_PATH = './dataset/list/cityscapes/val.lst'
IGNORE_LABEL = 255
BATCH_SIZE = 4
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
INPUT_SIZE = '769,769'
RESTORE_FROM = './deeplab_resnet.ckpt'
IMG_MAX_SIZE = 600

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--Output", type=str, default='./',
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="choose the number of recurrence.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="choose the number of recurrence.")
    parser.add_argument("--whole", type=bool, default=False,
                        help="use whole input size.")
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--datasets", type=str, default='cityscapes',
                    help="select the dataset to use. cityscapes and ade is available now.")
    return parser

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(net, image, tile_size, classes, recurrence):
    # interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = max(int(ceil((image_size[2] - tile_size[0]) / stride) + 1), 1)  # strided convolution formula
    tile_cols = max(int(ceil((image_size[3] - tile_size[1]) / stride) + 1), 1)
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[0], image_size[2], image_size[3], classes))
    count_predictions = np.zeros((1, image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            tile_counter += 1
            # print("Predicting tile %i" % tile_counter)
            padded_prediction = net(jt.Var(padded_img))
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            padded_prediction = nn.interpolate(padded_prediction, size=tile_size, mode='bilinear', align_corners=True).numpy().transpose(0,2,3,1)
            prediction = padded_prediction[0, 0:img.shape[2], 0:img.shape[3], :]
            count_predictions[0, y1:y2, x1:x2] += 1
            full_probs[:, y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    count_predictions = np.where(count_predictions > 0, count_predictions, 1)
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs

def predict_whole(net, image, tile_size, recurrence):
    N_, C_, H_, W_ = image.shape
    image = jt.Var(image)
    # interp = nn.Upsample(size=(H_, W_), mode='bilinear', align_corners=True)
    prediction = net(image)
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = nn.interpolate(prediction, size=(H_, W_), mode='bilinear', align_corners=True).numpy().transpose(0,2,3,1)
    return prediction

def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation, recurrence):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image = image.data
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((N_, H_, W_, classes))  
    for scale in scales:
        scale = float(scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)

        # stride = ceil(tile_size[0] * (1 - 1/3))
        # if scale_image.shape[2] < stride and scale_image.shape[3] < stride:
        #     scaled_probs = predict_whole(net, scale_image, tile_size, recurrence)
        # else:
        #     scaled_probs = predict_sliding(net, scale_image, tile_size, classes, recurrence)
        scaled_probs = predict_sliding(net, scale_image, tile_size, classes, recurrence)

        if flip_evaluation == True:
            # flip_scaled_probs = predict_whole(net, scale_image[:,:,:,::-1].copy(), tile_size, recurrence)
            flip_scaled_probs = predict_sliding(net, scale_image[:,:,:,::-1].copy(), tile_size, classes, recurrence)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:,::-1,:])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def main():
    """Create the model and start the evaluation process."""
    parser = get_parser()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        # cudnn.benchmark = True

        h, w = map(int, args.input_size.split(','))
        if args.whole:
            input_size = (1024, 2048)
        else:
            input_size = (h, w)

        seg_model = eval('networks.' + args.model + '.Seg_Model')(
            num_classes=args.num_classes, recurrence=args.recurrence
        )
        
        load_model(seg_model, args.restore_from)


        model = seg_model
        model.eval()
        if args.datasets == 'cityscapes':
            dataset = CSDataSet(args.data_dir, args.data_list, crop_size=input_size, mean=IMG_MEAN, scale=False, mirror=False)
        else:
            dataset = ADEDataSet(args.data_dir, args.data_list, crop_size=input_size, 
            mirror=False, mean=ADE_IMG_MEAN, std=ADE_IMG_STD,is_train=False, need_crop=False, imgSizes=600)
        test_loader = engine.get_test_loader(dataset)

        # if engine.distributed:
        #     test_sampler.set_epoch(0)

        data_list = []
        confusion_matrix = np.zeros((args.num_classes,args.num_classes))
        #palette = get_palette(256)
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        ade_palette_rgb = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]
        ade_palette = [i for k in ade_palette_rgb for i in k]

        save_path = os.path.join(os.path.dirname(args.Output), 'outputs')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(test_loader)//jt.world_size), file=sys.stdout,
                        bar_format=bar_format)
        dataloader = iter(test_loader)

        for idx in pbar:
            image, label, size, name = next(dataloader)
            size = size[0].numpy()
            with jt.no_grad():
                output = predict_multiscale(model, image, input_size, [1.0], args.num_classes, False, 0)


            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
            seg_gt = np.asarray(label.numpy()[:,:size[0],:size[1]], dtype=np.int)

            for i in range(image.size(0)): 
                output_im = PILImage.fromarray(seg_pred[i])
                if args.datasets == 'cityscapes':
                    output_im.putpalette(palette)
                elif args.datasets == 'ade':
                    output_im.putpalette(ade_palette)
                output_im.save(os.path.join(save_path, name[i]+'.png'))
        
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]
            # show_all(gt, output)
            confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)

            print_str = ' Iter{}/{}'.format(idx + 1, len(test_loader)//jt.world_size)
            pbar.set_description(print_str, refresh=False)

        #confusion_matrix = torch.from_numpy(confusion_matrix).contiguous().cuda()
        #confusion_matrix = engine.all_reduce_tensor(confusion_matrix, norm=False).cpu().numpy()
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()
        
        # getConfusionMatrixPlot(confusion_matrix)
        #if engine.distributed and engine.local_rank == 0:
        print({'meanIU':mean_IU, 'IU_array':IU_array})
        model_path = os.path.dirname(args.restore_from)
        with open(os.path.join(model_path, 'result.txt'), 'w') as f:
                f.write(json.dumps({'meanIU':mean_IU, 'IU_array':IU_array.tolist()}))

if __name__ == '__main__':
    main()
