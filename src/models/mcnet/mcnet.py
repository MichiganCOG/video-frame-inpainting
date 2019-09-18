import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from ...util.util import bgr2gray, inverse_transform, bgr2gray_batched


##################################################
##############  MC-NET PRIMITIVES  ###############
##################################################

class MotionEnc(nn.Module):
    """The motion encoder as defined by Villegas et al. (https://arxiv.org/abs/1706.08033).

    This module takes a difference frame and produces an encoded representation with reduced resolution. It also
    produces the intermediate convolutional activations for use with residual layers.
    """

    def __init__(self, gf_dim):
        """Constructor

        :param gf_dim: The number of filters in the first layer
        """
        super(MotionEnc, self).__init__()

        self.dyn_conv1 = nn.Sequential(
            nn.Conv2d(1, gf_dim, 5, padding=2),
            nn.ReLU()
        )

        self.dyn_conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim, gf_dim * 2, 5, padding=2),
            nn.ReLU()
        )

        self.dyn_conv3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim * 2, gf_dim * 4, 7, padding=3),
            nn.ReLU()
        )

        self.pool3 = nn.MaxPool2d(2)

    def forward(self, input_diff):
        """Forward method

        :param input_diff: A difference frame [batch_size, 1, h, w]
        :return: [batch_size, gf_dim*4, h/8, w/8]
        """
        dyn_conv1_out = self.dyn_conv1(input_diff)
        dyn_conv2_out = self.dyn_conv2(dyn_conv1_out)
        dyn_conv3_out = self.dyn_conv3(dyn_conv2_out)
        output = self.pool3(dyn_conv3_out)

        res_in = [dyn_conv1_out, dyn_conv2_out, dyn_conv3_out]

        return output, res_in


class ContentEnc(nn.Module):
    """The motion encoder as defined by Villegas et al. (https://arxiv.org/abs/1706.08033).

    This module takes a standard frame and produces an encoded representation with reduced resolution. It also
    produces the intermediate convolutional activations for use with residual layers.
    """

    def __init__(self, c_dim, gf_dim):
        """Constructor

        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param gf_dim: The number of filters in the first layer
        """

        super(ContentEnc, self).__init__()

        self.cont_conv1 = nn.Sequential(
            nn.Conv2d(c_dim, gf_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(gf_dim, gf_dim, 3, padding=1),
            nn.ReLU()
        )

        self.cont_conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim, gf_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(gf_dim * 2, gf_dim * 2, 3, padding=1),
            nn.ReLU()
        )

        self.cont_conv3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim * 2, gf_dim * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(gf_dim * 4, gf_dim * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(gf_dim * 4, gf_dim * 4, 3, padding=1),
            nn.ReLU()
        )

        self.pool3 = nn.MaxPool2d(2)

    def forward(self, raw):
        """Forward method

        :param raw: A raw image frame [batch_size, c_dim, h, w]
        :return: [batch_size, gf_dim*4, h/8, w/8]
        """
        cont_conv1_out = self.cont_conv1(raw)
        cont_conv2_out = self.cont_conv2(cont_conv1_out)
        cont_conv3_out = self.cont_conv3(cont_conv2_out)
        output = self.pool3(cont_conv3_out)

        res_in = [cont_conv1_out, cont_conv2_out, cont_conv3_out]

        return output, res_in


class CombLayers(nn.Module):
    """The combination layers as defined by Villegas et al. (https://arxiv.org/abs/1706.08033).

    This module combines the encoded representations of the past motion frames and the last content frame with
    convolutional layers.
    """

    def __init__(self, gf_dim):
        """Constructor

        :param gf_dim: The number of filters in the first layer
        """

        super(CombLayers, self).__init__()

        self.h_comb = nn.Sequential(
            nn.Conv2d(gf_dim * 8, gf_dim * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(gf_dim * 2, gf_dim * 4, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, h_dyn, h_cont):
        """Forward method

        :param h_dyn: The output from the MotionEnc module
        :param h_cont: The output from the ContentEnc module
        """
        input = torch.cat((h_dyn, h_cont), dim=1)
        return self.h_comb(input)


class Residual(nn.Module):
    """The residual layers as defined by Villegas et al. (https://arxiv.org/abs/1706.08033).

    This module combines a pair of "residual" convolutional activations from the MotionEnc and ContentEnc modules with
    convolutional layers.
    """

    def __init__(self, in_dim, out_dim):
        """Constructor

        :param in_dim: The number of channels in the input
        :param out_dim: The number of channels in the output
        """

        super(Residual, self).__init__()

        self.res = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1)
        )

    def forward(self, input_dyn, input_cont):
        """Forward method

        :param input_dyn: A set of intermediate activations from the MotionEnc module
        :param input_cont: A set of intermediate activations from the ContentEnc module
        """
        input = torch.cat((input_dyn, input_cont), dim=1)
        return self.res(input)


class DecCnn(nn.Module):
    """The decoder layers as defined by Villegas et al. (https://arxiv.org/abs/1706.08033).

    This module decodes the output of the CombLayers module into a full-resolution image. Optionally, it can incorporate
    activations from the Residual modules to help preserve spatial information.
    """

    def __init__(self, c_dim, gf_dim):
        """Constructor

        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param gf_dim: The number of filters in the first layer
        """

        super(DecCnn, self).__init__()

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim * 4, gf_dim * 4, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim * 4, gf_dim * 4, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, 3, padding=1),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim * 2, gf_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim * 2, gf_dim, 3, padding=1),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim, gf_dim, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim, c_dim, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, comb, res1, res2, res3):
        """Forward method

        :param comb: The output from the CombLayers module
        :param (res1, res2, res3): Outputs from each Residual module
        """

        dec3_out = self.dec3(self.fixed_unpooling(comb) + res3)
        dec2_out = self.dec2(self.fixed_unpooling(dec3_out) + res2)
        dec1_out = self.dec1(self.fixed_unpooling(dec2_out) + res1)

        return dec1_out

    def fixed_unpooling(self, x):
        """Unpools by spreading the values of x across a spaced-out grid. E.g.:

               x0x0x0
        xxx    000000
        xxx -> x0x0x0
        xxx    000000
               x0x0x0
               000000

        :param x: B x C x H x W FloatTensor Variable
        :return:
        """
        x = x.permute(0, 2, 3, 1)
        out = torch.cat((x, x.clone().zero_()), dim=3)
        out = torch.cat((out, out.clone().zero_()), dim=2)
        return out.view(x.size(0), 2*x.size(1), 2*x.size(2), x.size(3)).permute(0, 3, 1, 2)


class ConvLstmCell(nn.Module):
    """A convolutional LSTM cell (https://arxiv.org/abs/1506.04214)."""

    def __init__(self, feature_size, num_features, forget_bias=1, activation=F.tanh, bias=True):
        """Constructor

        :param feature_size: The kernel size of the convolutional layer
        :param num_features: Controls the number of input/output features of cell
        :param forget_bias: The bias for the forget gate
        :param activation: The activation function to use in the gates
        :param bias: Whether to use a bias for the convolutional layer
        """
        super(ConvLstmCell, self).__init__()

        self.feature_size = feature_size
        self.num_features = num_features
        self.forget_bias = forget_bias
        self.activation = activation

        self.conv = nn.Conv2d(num_features * 2, num_features * 4, feature_size, padding=(feature_size - 1) / 2,
                              bias=bias)

    def forward(self, input, state):
        """Forward method

        :param input: The current input to the ConvLSTM
        :param state: The previous state of the ConvLSTM (the concatenated memory cell and hidden state)
        """
        c, h = torch.chunk(state, 2, dim=1)
        conv_input = torch.cat((input, h) , dim=1)
        conv_output = self.conv(conv_input)
        (i, j, f, o) = torch.chunk(conv_output, 4, dim=1)
        new_c = c * F.sigmoid(f + self.forget_bias) + F.sigmoid(i) * self.activation(j)
        new_h = self.activation(new_c) * F.sigmoid(o)
        new_state = torch.cat((new_c, new_h), dim=1)
        return new_h, new_state


##################################################
####################  MC-NET  ####################
##################################################

class MCNetFillInModel(nn.Module):
    """A video frame inpainting network that only predicts the middle frames from the preceding frames."""

    def __init__(self, gf_dim, c_dim, feature_size, forget_bias=1, activation=F.tanh, bias=True):
        """Constructor

        :param gf_dim: The number of filters to use in the MotionEnc and ContentEnc modules
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param feature_size: The kernel size of the ConvLSTM
        :param forget_bias: The bias for the forget gate in the ConvLSTM
        :param activation: The activation function in the ConvLSTM
        :param bias: Whether to use a bias for the convolutional layer of the ConvLSTM
        """

        super(MCNetFillInModel, self).__init__()
        self.c_dim = c_dim
        self.conv_lstm_state_size = 8 * gf_dim

        self.generator = MCNet(gf_dim, c_dim, feature_size, forget_bias=forget_bias, activation=activation,
                               bias=bias)


    def forward(self, T, preceding_frames, following_frames):
        """Forward method

        :param T: The number of middle frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        """
        K = preceding_frames.size(1)

        # Get the content frames
        xt = preceding_frames[:, -1, :, :, :]

        # Compute forward difference frames
        gray_imgs_t_preceding = bgr2gray_batched(inverse_transform(preceding_frames)) if preceding_frames.size(2) > 1 else inverse_transform(preceding_frames)
        diff_in = gray_imgs_t_preceding[:, 1:, :, :, :] - gray_imgs_t_preceding[:, :-1, :, :, :]

        # Generate the forward and backward predictions
        forward_pred, _, _, _ = self.generator(K, T, diff_in, xt)

        # Stack time dimension (T x [B x C x H x W] -> [B x T x C x H x W]
        forward_pred = torch.stack(forward_pred, dim=1)

        return {
            'pred': forward_pred
        }


class MCNet(nn.Module):
    """The MC-Net video prediction network as defined by Villegas et al. (https://arxiv.org/abs/1706.08033)."""

    def __init__(self, gf_dim, c_dim, feature_size, forget_bias=1, activation=F.tanh, bias=True):
        """Constructor

        :param gf_dim: The number of filters to use in the MotionEnc and ContentEnc modules
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param feature_size: The kernel size of the ConvLSTM
        :param forget_bias: The forget bias to use in the ConvLSTM
        :param activation: The activation function to use in the ConvLSTM
        :param bias: Whether to use a bias for the ConvLSTM
        """
        super(MCNet, self).__init__()
        self.c_dim = c_dim
        self.gf_dim = gf_dim

        self.motion_enc = MotionEnc(gf_dim)
        self.conv_lstm_cell = ConvLstmCell(feature_size, 4 * gf_dim, forget_bias=forget_bias, activation=activation,
                                           bias=bias)
        self.content_enc = ContentEnc(c_dim, gf_dim)
        self.comb_layers = CombLayers(gf_dim)
        self.residual3 = Residual(gf_dim * 8, gf_dim * 4)
        self.residual2 = Residual(gf_dim * 4, gf_dim * 2)
        self.residual1 = Residual(gf_dim * 2, gf_dim * 1)
        self.dec_cnn = DecCnn(c_dim, gf_dim)


    def get_initial_conv_lstm_state(self, batch_size, image_size):
        """Get the initial state of the ConvLSTMCell.

        :param batch_size: The batch size of the input
        :param image_size: The resolution of the final video [H, W]
        """
        state = Variable(torch.zeros(batch_size, 8 * self.gf_dim, image_size[0]/8, image_size[1]/8),
                         requires_grad=False)
        if next(self.parameters()).is_cuda:
            state = state.cuda()
        return state


    def forward(self, K, T, diff_in, xt):
        """Forward method

        :param K: The number of past time steps
        :param T: The number of future time steps
        :param diff_in: The past difference frames [B x T x C x H x W]
        :param xt: The last past frame
        """

        # Split diff_in across the time dimension into an array
        diff_in = torch.chunk(diff_in, diff_in.shape[1], dim=1)
        diff_in = [x.squeeze(1) for x in diff_in]

        batch_size = xt.shape[0]
        image_size = xt.shape[2:4]
        state = self.get_initial_conv_lstm_state(batch_size, image_size)

        # Compute the motion encoding at each past time step
        for t in range(K-1):
            enc_h, res_m = self.motion_enc(diff_in[t])
            h_dyn, state = self.conv_lstm_cell(enc_h, state)

        # Keep track of outputs
        pred = []
        dyn = []
        cont = []
        res = []
        for t in range(T):
            # Compute the representation of the next motion frame
            if t > 0:
                enc_h, res_m = self.motion_enc(diff_in[-1])
                h_dyn, state = self.conv_lstm_cell(enc_h, state)
            # Compute the representation of the next content frame
            h_cont, res_c = self.content_enc(xt)
            # Combine the motion and content encodings
            h_tpl = self.comb_layers(h_dyn, h_cont)
            # Store the motion and content encodings
            dyn.append(h_dyn)
            cont.append(h_cont)
            # Pass intermediate activations through the residual layers
            res_1 = self.residual1(res_m[0], res_c[0])
            res_2 = self.residual2(res_m[1], res_c[1])
            res_3 = self.residual3(res_m[2], res_c[2])
            res.append([res_1, res_2, res_3])
            # Pass activations through the decoder
            x_hat = self.dec_cnn(h_tpl, res_1, res_2, res_3)

            # Obtain grayscale versions of the predicted frames
            if self.c_dim == 3:
                x_hat_gray = bgr2gray(inverse_transform(x_hat))
                xt_gray = bgr2gray(inverse_transform(xt))
            else:
                x_hat_gray = inverse_transform(x_hat)
                xt_gray = inverse_transform(xt)

            # Compute the next MotionEnc input, which is difference between grayscale frames
            diff_in.append(x_hat_gray - xt_gray)
            # Update last past frame
            xt = x_hat
            # Update outputs
            pred.append(x_hat.view(-1, self.c_dim, image_size[0], image_size[1]))

        return pred, dyn, cont, res


