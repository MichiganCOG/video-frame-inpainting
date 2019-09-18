import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from ..mcnet.mcnet import MCNet
from ...util.util import bgr2gray_batched, inverse_transform


class BidirectionalTimeWeightedAverageFillInModel(nn.Module):
    """This predicts the middle frames by making the forward and backward predictions, then taking a weighted
    average between corresponding frames.

    For middle frame t (1-indexed) within T middle frames, the weight of the backward stream is t/(T+1); the
    weight of the forward stream is 1 minus that.
    """

    def __init__(self, gf_dim, c_dim, feature_size, forget_bias=1, activation=F.tanh, bias=True):
        super(BidirectionalTimeWeightedAverageFillInModel, self).__init__()
        self.c_dim = c_dim
        self.conv_lstm_state_size = 8 * gf_dim

        self.generator = MCNet(gf_dim, c_dim, feature_size, forget_bias=forget_bias, activation=activation,
                               bias=bias)

    def forward(self, T, preceding_frames, following_frames):
        K = preceding_frames.size(1)
        F = following_frames.size(1)

        # Get the content frames
        xt = preceding_frames[:, -1, :, :, :]
        xt_F = following_frames[:, 0, :, :, :]

        # Compute forward difference frames
        gray_imgs_t_preceding = bgr2gray_batched(inverse_transform(preceding_frames)) if preceding_frames.size(2) > 1 else inverse_transform(preceding_frames)
        diff_in = gray_imgs_t_preceding[:, 1:, :, :, :] - gray_imgs_t_preceding[:, :-1, :, :, :]

        # Compute backward difference frames
        gray_imgs_t_following = bgr2gray_batched(inverse_transform(following_frames)) if following_frames.size(2) > 1 else inverse_transform(following_frames)
        rev_indexes = Variable(torch.LongTensor(range(F-1, -1, -1))).cuda()
        rev = gray_imgs_t_following.index_select(1, rev_indexes)
        diff_in_F = rev[:, 1:, :, :, :] - rev[:, :-1, :, :, :]

        # Generate the forward and backward predictions
        forward_pred, forward_dyn, forward_cont, forward_res = self.generator(K, T, diff_in, xt)
        backward_pred, backward_dyn, backward_cont, backward_res = self.generator(F, T, diff_in_F, xt_F)
        # Correct the order of the backward frames
        backward_pred = backward_pred[::-1]

        # Store the final predictions
        combination = []
        # Compute weight of backward prediction per time step
        w = np.linspace(0, 1, num=T+2).tolist()[1:-1]
        for t in range(T):
            # Merge the forward and backward predictions
            combination.append((1 - w[t]) * forward_pred[t] + w[t] * backward_pred[t])

        # Stack time dimension (T x [B x C x H x W] -> [B x T x C x H x W]
        combination = torch.stack(combination, dim=1)
        forward_pred = torch.stack(forward_pred, dim=1)
        backward_pred = torch.stack(backward_pred, dim=1)

        return {
            'pred': combination,
            'pred_forward': forward_pred,
            'pred_backward': backward_pred
        }