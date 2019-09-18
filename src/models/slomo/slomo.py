from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import cv2
import numpy as np
import logging
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image
import sys
import os
sys.path.insert(
    0, os.path.realpath(os.path.dirname(__file__) + "/../../.."))
from src.data.base_dataset import ContiguousVideoClipDataset
from src.util.util import makedir, listopt, to_numpy, inverse_transform


##################################################
##############  SloMo PRIMITIVES  ###############
##################################################

class Encoder(nn.Module):

    def __init__(self, gf_dim, input_dim, alpha=0.1):
        """Constructor
        
        Parameters:
            gf_dim: The number of filters in the first layer
            input_dim: dimension of input
        """

        super(Encoder, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(input_dim, gf_dim, 7, padding=3),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim, gf_dim, 7, padding=3),
            nn.LeakyReLU(alpha)
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim, gf_dim * 2, 5, padding=2),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 2, gf_dim * 2, 5, padding=2),
            nn.LeakyReLU(alpha)
        )

        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim * 2, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 4, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim * 4, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 8, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.enc5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim * 8, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 16, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.enc6 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim * 16, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 16, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

    def forward(self, input_imgs):

        enc1_out = self.enc1(input_imgs)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)
        output = self.enc6(enc5_out)

        res_in = [enc1_out, enc2_out, enc3_out, enc4_out, enc5_out]

        return output, res_in


class ComputeDecoder(nn.Module):

    def __init__(self, gf_dim, out_dim, alpha=0.1):
        """Constructor
        
        Parameters:
            gf_dim: The number of filters in the first layer
            out_dim: The dimension of output
        """

        super(ComputeDecoder, self).__init__()

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec1 = nn.Sequential(
            nn.Conv2d(gf_dim * 32, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 16, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec2 = nn.Sequential(
            nn.Conv2d(gf_dim * 16, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 8, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec3 = nn.Sequential(
            nn.Conv2d(gf_dim * 8, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample4 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec4 = nn.Sequential(
            nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 2, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample5 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec5 = nn.Sequential(
            nn.Conv2d(gf_dim * 2, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.output = nn.Conv2d(gf_dim, out_dim, 1)
        self.tanh = nn.Tanh()


    def forward(self, encoded_input, res_in):

        upsample1_out = self.upsample1(encoded_input)
        dec1_out = self.dec1(
            torch.cat((upsample1_out, res_in[-1]), 1))
        upsample2_out = self.upsample2(dec1_out)
        dec2_out = self.dec2(torch.cat((upsample2_out, res_in[-2]), 1))
        upsample3_out = self.upsample3(dec2_out)
        dec3_out = self.dec3(torch.cat((upsample3_out, res_in[-3]), 1))
        upsample4_out = self.upsample4(dec3_out)
        dec4_out = self.dec4(torch.cat((upsample4_out, res_in[-4]), 1))
        upsample5_out = self.upsample5(dec4_out)
        dec5_out = self.dec5(torch.cat((upsample5_out, res_in[-5]), 1))
        output = self.output(dec5_out)
        output = self.tanh(output)

        return output


class RefineDecoder(nn.Module):

    def __init__(self, gf_dim, out_dim, alpha=0.1):
        """Constructor
        
        Parameters:
            gf_dim: The number of filters in the first layer
            out_dim: The dimension of output
        """

        super(RefineDecoder, self).__init__()

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec1 = nn.Sequential(
            nn.Conv2d(gf_dim * 32, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 16, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec2 = nn.Sequential(
            nn.Conv2d(gf_dim * 16, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 8, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec3 = nn.Sequential(
            nn.Conv2d(gf_dim * 8, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample4 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec4 = nn.Sequential(
            nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 2, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample5 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec5 = nn.Sequential(
            nn.Conv2d(gf_dim * 2, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.output = nn.Conv2d(gf_dim, out_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, encoded_input, res_in):

        upsample1_out = self.upsample1(encoded_input)
        dec1_out = self.dec1(
            torch.cat((upsample1_out, res_in[-1]), 1))
        upsample2_out = self.upsample2(dec1_out)
        dec2_out = self.dec2(torch.cat((upsample2_out, res_in[-2]), 1))
        upsample3_out = self.upsample3(dec2_out)
        dec3_out = self.dec3(torch.cat((upsample3_out, res_in[-3]), 1))
        upsample4_out = self.upsample4(dec3_out)
        dec4_out = self.dec4(torch.cat((upsample4_out, res_in[-4]), 1))
        upsample5_out = self.upsample5(dec4_out)
        dec5_out = self.dec5(torch.cat((upsample5_out, res_in[-5]), 1))
        output = self.output(dec5_out)

        delta_F_t_0, delta_F_t_1, V_t_0 = torch.split(output, 2, dim=1)
        V_t_0 = self.sigmoid(V_t_0)
        delta_F_t_0 = self.tanh(delta_F_t_0)
        delta_F_t_1 = self.tanh(delta_F_t_1)

        return delta_F_t_0, delta_F_t_1, V_t_0


class FlowWarper(nn.Module):

    def forward(self, img, uv):

        super(FlowWarper, self).__init__()
        H = int(img.shape[-2])
        W = int(img.shape[-1])
        x = np.arange(0, W)
        y = np.arange(0, H)
        gx, gy = np.meshgrid(x, y)
        grid_x = Variable(torch.Tensor(gx), requires_grad=False).cuda()
        grid_y = Variable(torch.Tensor(gy), requires_grad=False).cuda()
        u = uv[:, 0, :, :]
        v = uv[:, 1, :, :]
        X = grid_x.unsqueeze(0) + u
        Y = grid_y.unsqueeze(0) + v
        X = 2 * (X / W - 0.5)
        Y = 2 * (Y / H - 0.5)
        grid_tf = torch.stack((X, Y), dim=3)
        img_tf = F.grid_sample(img, grid_tf)

        return img_tf


class SloMo(nn.Module):
    """The SloMo video prediction network. """

    def __init__(self, gf_dim, c_input_dim):

        super(SloMo, self).__init__()
        self.c_input_dim = c_input_dim
        self.compute_enc = Encoder(gf_dim, 2 * c_input_dim)
        self.compute_dec = ComputeDecoder(gf_dim, 4)
        self.flow_warper = FlowWarper()
        self.refine_enc = Encoder(gf_dim, 4 * c_input_dim + 4)
        self.refine_dec = RefineDecoder(gf_dim, 5)

    def forward(self, T, I0, I1):

        # I0 and I1 have the format [batch_size, channel, W, H]
        img = torch.cat((I0, I1), 1)
        compute_enc_out, compute_res_in = self.compute_enc(img)
        compute_dec_out = self.compute_dec(compute_enc_out, compute_res_in)
        F_0_1 = compute_dec_out[:, :2, :, :]
        F_1_0 = compute_dec_out[:, 2:, :, :]
        first = True
        for t_ in range(T):
            t = (t_ + 1) / (T + 1)
            F_t_0 = -(1 - t) * t * F_0_1 + t ** 2 * F_1_0
            F_t_1 = (1 - t) * (1 - t) * F_0_1 - t * (1 - t) * F_1_0
            g_I0_F_t_0 = self.flow_warper(img[:, :self.c_input_dim, :, :], F_t_0)
            g_I1_F_t_1 = self.flow_warper(img[:, self.c_input_dim:, :, :], F_t_1)
            interp_input = torch.cat((I0, g_I0_F_t_0, F_t_0, F_t_1, g_I1_F_t_1, I1), 1)
            interp_enc_out, interp_res_in = self.refine_enc(interp_input)
            delta_F_t_0, delta_F_t_1, V_t_0 = self.refine_dec(interp_enc_out, interp_res_in)
            F_t_0_refine = delta_F_t_0 + F_t_0
            F_t_0_refine = torch.clamp(F_t_0_refine, min=-1, max=1)
            F_t_1_refine = delta_F_t_1 + F_t_1
            F_t_1_refine = torch.clamp(F_t_1_refine, min=-1, max=1)
            V_t_1 = 1 - V_t_0
            g_I0_F_t_0_refine = self.flow_warper(I0, F_t_0_refine)
            g_I1_F_t_1_refine = self.flow_warper(I1, F_t_1_refine)
            normalization = (1 - t) * V_t_0 + t * V_t_1
            interp_image = ((1 - t) * V_t_0 * g_I0_F_t_0_refine + t * V_t_1 * g_I1_F_t_1_refine) / normalization
            F_t_0 = torch.unsqueeze(F_t_0, 1)
            F_t_1 = torch.unsqueeze(F_t_1, 1)
            interp_image = torch.unsqueeze(interp_image, 1)
            if first:
                predictions = interp_image
                F_t_0_collector = F_t_0
                F_t_1_collector = F_t_1
                first = False
            else:
                F_t_0_collector = torch.cat((F_t_0, F_t_0_collector), 1)
                F_t_1_collector = torch.cat((F_t_1, F_t_1_collector), 1)
                predictions = torch.cat((interp_image, predictions), 1)

        return predictions, F_0_1, F_1_0, F_t_0_collector, F_t_1_collector


class SloMoFillInModel(nn.Module):

    def __init__(self, gf_dim=32, c_input_dim=3):

        super(SloMoFillInModel, self).__init__()

        self.generator = SloMo(gf_dim, c_input_dim)


    def forward(self, T, preceding_frames, following_frames):
        """Forward method

        :param T: The number of middle frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        """

        # Generate the forward and backward predictions
        pred, F_0_1, F_1_0, F_t_0_collector, F_t_1_collector = self.generator(T, preceding_frames[:, -1, :, :, :], following_frames[:, 0, :, :, :])

        return {
            'pred': pred,
            'F_0_1': F_0_1,
            'F_1_0': F_1_0,
            'F_t_0_collector': F_t_0_collector,
            'F_t_1_collector': F_t_1_collector
        }
