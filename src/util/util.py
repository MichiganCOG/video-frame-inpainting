import io
import os
import sys
from copy import deepcopy
from Queue import Queue

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.nn import init
from torch.autograd import Variable

from ..discriminators.SNDiscriminator import SNConv2d, SNLinear


def inverse_transform(images):
    return (images+1.)/2


def fore_transform(images):
    return images * 2 - 1


def bgr2gray(image):
    # rgb -> grayscale 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray_ = 0.1140 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.2989 * image[:, 2, :, :]
    gray = torch.unsqueeze(gray_, 1)
    return gray


def bgr2gray_batched(image):
    # rgb -> grayscale 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray_ = 0.1140 * image[:, :, 0, :, :] + 0.5870 * image[:, :, 1, :, :] + 0.2989 * image[:, :, 2, :, :]
    gray = torch.unsqueeze(gray_, 2)
    return gray


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def draw_frame_tensor(img, K, T):
    """ Draws a red or green frame on the boundary of the given image tensor
    :param img: a torch.tensor image with all frames of a test case (B x T x C x H x W)
    :param K: the number of input frames
    :param T: the number of output frames
    :return: a torch,tensor image
    """

    # Make borders black
    img[:, :, :, :2, :] = 0
    img[:, :, :, :, :2] = 0
    img[:, :, :, -2:, :] = 0
    img[:, :, :, :, -2:] = 0

    # Maximize green channel on borders for preceding and following frames
    img[:, :K, 1, :2, :] = img[:, K+T:, 1, :2, :] = 1
    img[:, :K, 1, :, :2] = img[:, K+T:, 1, :, :2] = 1
    img[:, :K, 1, -2:, :] = img[:, K+T:, 1, -2:, :] = 1
    img[:, :K, 1, :, -2:] = img[:, K+T:, 1, :, -2:] = 1

    # Maximize red channel on borders for middle frames
    img[:, K:K+T, 0, :2, :] = 1
    img[:, K:K+T, 0, :, :2] = 1
    img[:, K:K+T, 0, -2:, :] = 1
    img[:, K:K+T, 0, :, -2:] = 1

    return img


def draw_err_plot(err,  err_name, lims, path=None):
    """Draws an average PSNR or SSIM error plot and either saves it to disk or returns the image as a np.ndarray.

    :param err: The error values to plot as a N x T np.ndarray
    :param err_name: The title of the plot
    :param lims: The axis limits of the plot
    :param path: The path to write the plot image to. If None, return the plot image as an np.ndarray
    """
    avg_err = np.mean(err, axis=0)
    T = err.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(1, T+1)
    ax.plot(x, avg_err, marker='d')
    ax.set_xlabel('time steps')
    ax.set_ylabel(err_name)
    ax.grid()
    ax.set_xticks(x)
    ax.axis(lims)
    if path is None:
        plot_buf = gen_plot(fig)
        im = np.array(Image.open(plot_buf), dtype=np.uint8)
        plt.close(fig)
        return im
    else:
        plt.savefig(path)


def gen_plot(fig):
    """
    Create a pyplot plot and save to buffer.
    https://stackoverflow.com/a/38676842
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf


def visual_grid(visuals, K, T):
    """Generate the qualitative results for validation

    :param visuals: A dictionary containing the data to visualize. The values must be BGR frames given as a
                    B x T x C x H x W tensor.
    :param K: int, the number of input frames
    :param T: int, the number of output frames
    :return: a torch.tensor with the visualization of qualitative results for validation
    """

    # B x T x C x H x W
    batch_size, seq_len, c_dim, H, W = next(visuals.itervalues()).shape

    vis_seqs = []
    for vis_seq_label, vis_seq in visuals.iteritems():
        # Map colors from [-1, 1] to [0, 1]
        vis_seq = inverse_transform(vis_seq)
        if c_dim == 1:
            # If images are grayscale, add extra color channels so colored borders can be added
            vis_seq = torch.cat([vis_seq] * 3, dim=2)
        else:
            # Swap from BGR to RGB
            rev_indexes = torch.arange(c_dim-1, -1, step=-1).long()
            vis_seq = vis_seq.index_select(2, as_variable(rev_indexes).cuda())
        # Draw colored borders
        bordered_vis_seq = draw_frame_tensor(vis_seq, K, T)
        # Add to sequences to visualize
        vis_seqs.append(bordered_vis_seq)

    # Group all visualizations by batch
    grouped_vis_seqs = torch.stack(vis_seqs, dim=1)
    # Combine all dimensions except for color, height, and width
    grouped_vis_seqs = grouped_vis_seqs.view(batch_size * seq_len * len(vis_seqs), 3, H, W)

    # Create the grid of images to visualize
    grid = vutils.make_grid(grouped_vis_seqs.data, nrow=seq_len)
    grid = torch.clamp(grid, 0, 1)

    return grid


def listopt(opt, f=sys.stdout):
    """Pretty-print a given namespace either to console or to a file.

    :param opt: A namespace
    :param f: The file descriptor to write to
    """
    args = vars(opt)

    f.write('------------ Options -------------\n')
    for k, v in sorted(args.items()):
        f.write('%s: %s\n' % (str(k), str(v)))
    f.write('-------------- End ----------------\n')


def to_numpy(tensor, transpose=None):
    """Converts the given Tensor to a np.ndarray.

    :param tensor: The Tensor to convert
    :param transpose: An iterable specifying the new dimensional ordering
    """

    if tensor.is_cuda:
        tensor = tensor.cpu()
    arr = tensor.numpy()
    if transpose is not None:
        arr = np.transpose(arr, transpose)

    return arr


def move_to_devices(model):
    """Moves the model to the first available GPU."""
    return model.cuda()


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, SNConv2d):
        init.xavier_normal(m.weight.data, gain=1)
        init.constant(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear) or isinstance(m, SNLinear):
        init.uniform(m.weight.data, 0.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def inf_data_loader(data_loader):
    """A generator that infinitely loops through the batches in the given data loader."""
    while True:
        for data in data_loader:
            yield data


def get_folder_paths_at_depth(root_path, depth):
    """Collect the path to all folders that are exactly `depth` levels under the given path (direct children are
    at level 1).

    :param root_path: The path to start at
    :param depth: The depth to extract children from
    """

    assert(depth >= 0)
    q = Queue()
    q.put((root_path, 0))
    ret = []
    while not q.empty():
        path, cur_depth = q.get()
        if os.path.isdir(path):
            if cur_depth == depth:
                ret.append(path)
            else:
                for child in os.listdir(path):
                    q.put((os.path.join(path, child), cur_depth+1))
    return ret


def as_variable(tensor):
    """Convert a torch Tensor into a Variable

    :param tensor: The Tensor to convert
    :return: Variable wrapping the given Tensor
    """
    return Variable(tensor, requires_grad=False)


def module_is_cuda(module):
    """Returns whether this module has been cudafied

    :param module: The module to check for cudafication
    :return: bool
    """
    return next(module.parameters()).is_cuda


def merge_dicts(*dicts):
    """Merge the given dictionaries. Note that the first dictionary gets mutated by this function.

    :param *dicts: The dictionaries to merge
    :return: A dict whose keys and values are the union of the keys and values from the given dictionaries
    """
    ret = dicts[0]
    for cur_dict in dicts[1:]:
        for k, v in cur_dict.iteritems():
            ret[k] = v

    return ret

def dict_to_markdown_table_str(d):
    sorted_keys = sorted(d.keys())

    lines = ['Key | Value', '--- | ---']
    for key in sorted_keys:
        key_value_str = '%s | %s' % (key, str(d[key]))
        lines.append(key_value_str)

    return '\n'.join(lines)
