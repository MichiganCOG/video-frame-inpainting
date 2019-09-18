import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from ...separable_convolution.SeparableConvolution import SeparableConvolution
from ...util.util import bgr2gray_batched, inverse_transform
from ..mcnet.mcnet import Residual, MCNet
from ..tai.tai import create_encoder_blocks, create_decoder_blocks, create_1d_kernel_generator_block


class TimeWeightedInterpolationFillInModel(nn.Module):
    """This predicts the middle frames with bidirectional prediction, then time-agnostic NN interpolation,
    then time-weighted averaging.

    For middle frame t (1-indexed) within T middle frames, the weight of the first following frame is t/(T+1); the
    weight of the last preceding frame is 1 minus that.
    """

    def __init__(self, gf_dim, c_dim, feature_size, ks, num_block=5, kf_dim=32, layers=3, forget_bias=1,
                 activation=F.tanh, bias=True):
        """Constructor

        :param gf_dim: The number of filters to use in the MotionEnc and ContentEnc modules
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param feature_size: The kernel size of the ConvLSTM
        :param ks: The size of the 1D kernel to generate with the KernelNet module
        :param num_block: Controls the number of blocks to use in the encoder and decoder chains of the KernelNet module
        :param layers: The number of layers to use in each encoder and decoder block in the KernelNet module
        :param kf_dim: Controls the number of filters in each encoder and decoder block in the KernelNet module
        :param forget_bias: The bias for the forget gate in the ConvLSTM
        :param activation: The activation function in the ConvLSTM
        :param bias: Whether to use a bias for the convolutional layer of the ConvLSTM
        """

        super(TimeWeightedInterpolationFillInModel, self).__init__()

        self.conv_lstm_state_size = 8 * gf_dim

        self.mcnet = MCNet(gf_dim, c_dim, feature_size, forget_bias=forget_bias, activation=activation, bias=bias)

        self.merge_residual3 = Residual(gf_dim * 8, kf_dim * 4)
        self.merge_residual2 = Residual(gf_dim * 4, kf_dim * 2)
        self.merge_residual1 = Residual(gf_dim * 2, kf_dim * 1)

        self.interp_net = TWI(gf_dim, ks, num_block, layers, kf_dim)


    def forward(self, T, preceding_frames, following_frames):
        """Forward method

        :param T: The number of middle frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        """
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
        forward_pred, forward_dyn, forward_cont, forward_res = self.mcnet(K, T, diff_in, xt)
        backward_pred, backward_dyn, backward_cont, backward_res = self.mcnet(F, T, diff_in_F, xt_F)
        # Correct the order of the backward frames
        backward_pred = backward_pred[::-1]
        backward_dyn = backward_dyn[::-1]
        backward_cont = backward_cont[::-1]
        backward_res = backward_res[::-1]

        # Store the final predictions and pre-summed outputs of the interpolation network
        combination = []
        interp_net_outputs_1 = []
        interp_net_outputs_2 = []
        # Compute weight of backward prediction per time step
        w = np.linspace(0, 1, num=T+2).tolist()[1:-1]
        for t in range(T):
            merged_res = []
            merged_res.append(self.merge_residual1(forward_res[t][0], backward_res[t][0]))
            merged_res.append(self.merge_residual2(forward_res[t][1], backward_res[t][1]))
            merged_res.append(self.merge_residual3(forward_res[t][2], backward_res[t][2]))

            variableDot1, variableDot2 = self.interp_net(forward_pred[t], backward_pred[t], forward_dyn[t],
                                                         backward_dyn[t], forward_cont[t], backward_cont[t],
                                                         merged_res)

            # Store the pre-summed outputs of the interpolation network
            interp_net_outputs_1.append(variableDot1)
            interp_net_outputs_2.append(variableDot2)
            # Merge the modified forward and backward predictions
            combination.append((1 - w[t]) * variableDot1 + w[t] * variableDot2)

        # Stack time dimension (T x [B x C x H x W] -> [B x T x C x H x W]
        combination = torch.stack(combination, dim=1)
        forward_pred = torch.stack(forward_pred, dim=1)
        backward_pred = torch.stack(backward_pred, dim=1)
        interp_net_outputs_1 = torch.stack(interp_net_outputs_1, dim=1)
        interp_net_outputs_2 = torch.stack(interp_net_outputs_2, dim=1)

        return {
            'pred': combination,
            'pred_forward': forward_pred,
            'pred_backward': backward_pred,
            'interp_net_outputs_1': interp_net_outputs_1,
            'interp_net_outputs_2': interp_net_outputs_2
        }


class TWI(nn.Module):
    """Time-Weighted Interpolation (TWI) module

    An adapted version of the adaptive separable convolution network for video frame interpolation as proposed by
    Niklaus et al. (https://arxiv.org/abs/1708.01692).

    Instead of taking raw frames, this module takes encoded representations of frames at a reduced resolution, as well
    as intermediate activations associated with the encoded representations.

    Unlike TAI, this module CANNOT take in the ratio information to distinguish the relative importance of the two
    input frames.
    """

    def __init__(self, gf_dim, ks, num_block, layers, kf_dim):
        """Constructor

        :param gf_dim: The number of channels in the input encodings
        :param ks: The size of the 1D kernel
        :param num_block: Controls the number of blocks to use in the encoder and decoder chains
        :param layers: The number of layers to use in each encoder and decoder block
        :param kf_dim: Controls the number of filters in each encoder and decoder block
        """

        super(TWI, self).__init__()

        assert layers >= 1, 'layers in per block should be no smaller than 1, but layers=[%d]' % layers
        assert num_block >= 4, '# blocks should be no less than 3, but num_block=%d' % num_block

        self.kf_dim = kf_dim
        self.ks = ks
        self.layers = layers
        self.num_block = num_block

        # Create the chain of encoder blocks
        moduleConv, modulePool = create_encoder_blocks(3, num_block, layers, gf_dim * 8 * 2, kf_dim)
        self.moduleConv = torch.nn.ModuleList(moduleConv)
        self.modulePool = torch.nn.ModuleList(modulePool)

        # Create the chain of decoder blocks
        moduleDeconv, moduleUpsample = create_decoder_blocks(num_block - 1, kf_dim, layers, -1)
        self.moduleDeconv = torch.nn.ModuleList(moduleDeconv)
        self.moduleUpsample = torch.nn.ModuleList(moduleUpsample)

        # Create the adaptive kernel blocks
        self.moduleVertical1 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)
        self.moduleVertical2 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)
        self.moduleHorizontal1 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)
        self.moduleHorizontal2 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)

        self.modulePad = torch.nn.ReplicationPad2d([int(math.floor(self.ks / 2.0)), int(math.floor(self.ks / 2.0)),
                                                    int(math.floor(self.ks / 2.0)), int(math.floor(self.ks / 2.0))])
        self.separableConvolution = SeparableConvolution.apply


    def forward(self, variableInput1, variableInput2, variableDyn1, variableDyn2, variableCont1, variableCont2,
                variableRes):
        """Forward method

        :param variableInput1: The encoding of the first frame to interpolate between
        :param variableInput2: The encoding of the second frame to interpolate between
        :param variableDyn1: The intermediate activations from MotionEnc associated with the first frame
        :param variableDyn2: The intermediate activations from MotionEnc associated with the second frame
        :param variableCont1: The intermediate activations from ContentEnc associated with the first frame
        :param variableCont2: The intermediate activations from ContentEnc associated with the second frame
        :param variableRes: The output of the residual layers that combine the residual activations from both frames
        """

        variableJoin = torch.cat([variableDyn1, variableDyn2, variableCont1, variableCont2], 1)

        variableConv = []
        variablePool = []

        # Pass the input through the encoder chain
        for i in range(self.num_block-3):
            if i == 0:
                variableConv.append(self.moduleConv[i](variableJoin))
            else:
                variableConv.append(self.moduleConv[i](variablePool[-1]))
            variablePool.append(self.modulePool[i](variableConv[-1]))

        # Pass the result through the decoder chain, applying skip connections from the encoder
        variableDeconv = []
        variableUpsample = []
        variableCombine = []
        for i in range(self.num_block-1):
            if i == 0:
                layer_input = variablePool[-1]
            else:
                layer_input = variableCombine[-1]
            variableDeconv.append(self.moduleDeconv[i](layer_input))

            # Upsample
            variableUpsample.append(self.moduleUpsample[i](variableDeconv[-1]))
            if i < (self.num_block-3):
                # Apply skip connection from the encoder
                variableCombine.append(variableUpsample[-1] + variableConv[self.num_block-3-i-1])
            else:
                # Apply skip connection from the residual layers
                variableCombine.append(variableUpsample[-1] + variableRes[self.num_block-i-1])

        # Apply the kernels to the source images
        variableDot1 = self.separableConvolution(self.modulePad(variableInput1),
                                                 self.moduleVertical1(variableCombine[-1]),
                                                 self.moduleHorizontal1(variableCombine[-1]),
                                                 self.ks)
        variableDot2 = self.separableConvolution(self.modulePad(variableInput2),
                                                 self.moduleVertical2(variableCombine[-1]),
                                                 self.moduleHorizontal2(variableCombine[-1]),
                                                 self.ks)
        return variableDot1, variableDot2