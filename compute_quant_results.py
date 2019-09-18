import argparse
import os

import numpy as np
from PIL import Image
from skimage.measure import compare_ssim, compare_psnr
import torchvision
import torch
from torchvision.transforms.functional import to_tensor
from torch.autograd import Variable

from src.util.util import get_folder_paths_at_depth, makedir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('qual_results_root', type=str)
    parser.add_argument('quant_results_root', type=str)
    parser.add_argument('K', type=int, help='Number of preceding frames')
    parser.add_argument('T', type=int, help='Number of middle frames')
    parser.add_argument('--depth', type=int, default=1,
                        help='Depth of the folders for each video (e.g. 2 for <qual_results_root>/<action>/<video>)')
    args = parser.parse_args()

    # Get the paths to the qualitative frames of each test video
    qual_frame_root_paths = get_folder_paths_at_depth(args.qual_results_root, args.depth)

    if len(qual_frame_root_paths) == 0:
        print('Failed to find any qualitative results (make sure you ran predict.py before this script). Quitting...')
        return

    print('Now computing quantitative results...')

    psnr_table = np.zeros((len(qual_frame_root_paths), args.T))
    ssim_table = np.zeros((len(qual_frame_root_paths), args.T))
    video_list = []

    for i, qual_frame_root_path in enumerate(qual_frame_root_paths):
        video_list.append(qual_frame_root_path)
        for t in xrange(args.K, args.K + args.T):
            try:
                gt_middle_frame = Image.open(os.path.join(qual_frame_root_path, 'gt_middle_%04d.png' % t))
            except IOError:
                raise RuntimeError('Failed to find GT middle frame at %s (did you generate GT middle frames and use '
                                   'the right values for K and T?)'
                                   % os.path.join(qual_frame_root_path, 'gt_middle_%04d.png' % t))
            pred_middle_frame = Image.open(os.path.join(qual_frame_root_path, 'pred_middle_%04d.png' % t))
            psnr_table[i, t - args.K] = compare_psnr(np.array(pred_middle_frame), np.array(gt_middle_frame))
            ssim_table[i, t - args.K] = compare_ssim(np.array(gt_middle_frame), np.array(pred_middle_frame),
                                                     multichannel=(gt_middle_frame.mode == 'RGB'))

    # Save PSNR and SSIM tables and video list to a file
    makedir(args.quant_results_root)
    video_list = np.array(video_list)

    np.savez(os.path.join(args.quant_results_root, 'results.npz'),
             psnr=psnr_table,
             ssim=ssim_table,
             video=video_list)

    print('Done computing quantitative results.')


def compare_perceptual_score(X, Y):
    if X.shape != Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16_conv = torch.nn.Sequential(*list(torch.nn.Sequential(*list(vgg16.children())[0]))[:22]).cuda()
    for param in vgg16_conv.parameters():
        param.requires_grad = False
    MSE_loss = torch.nn.MSELoss(size_average=True, reduce=True)

    # perceptual loss
    vgg_16_X = vgg16_conv(X)
    vgg_16_Y = vgg16_conv(Y)
    perceptual_loss = MSE_loss(vgg_16_X, vgg_16_Y)
    perceptual_score = 1 / (1 + perceptual_loss.data[0])
    return perceptual_score


if __name__ == '__main__':
    main()
