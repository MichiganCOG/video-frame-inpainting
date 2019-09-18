from abc import ABCMeta, abstractmethod

import torch
from torch import nn as nn

from src.models.self_attention.submodules import ConvTransformerEncoder, ConvTransformerDecoder, \
    ConvSkipConTransformerEncoder, ConvSkipConTransformerDecoder, ConvHiddenTransformerDecoder
from ...util.util import as_variable, module_is_cuda


class BaseSCTSkipConFillInModel(nn.Module):

    __metaclass__ = ABCMeta

    def __init__(self, C, num_blocks, num_heads, d_v, d_ff):
        """Constructor

        :param num_blocks: Number of blocks in the encoder stack
        :param num_heads: Number of heads to use for each multi-head attention module
        :param d_v: Dimensionality (number of features) of the values
        :param d_ff: Intermediate dimensionality of the "point-wise" feed-forward layers
        :param C: Number of input channels
        """

        super(BaseSCTSkipConFillInModel, self).__init__()

        self.d_v = d_v
        self.C = C

        self.frame_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C, d_v // 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(d_v // 8, d_v // 8, 3, padding=1)
            ),  # B x T x d_v/8 x H x W
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(d_v // 8, d_v // 4, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(d_v // 4, d_v // 4, 3, padding=1)
            ),  # B x T x d_v/4 x H/2 x W/2
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(d_v // 4, d_v // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(d_v // 2, d_v // 2, 3, padding=1)
            ),  # B x T x d_v/2 x H/4 x W/4
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(d_v // 2, d_v, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(d_v, d_v, 3, padding=1)
            )  # B x T x d_v x H/8 x W/8
        ])

        self.frame_decoder = nn.ModuleList([
            # B x T x d_v x H/8 x W/8
            nn.Sequential(
                nn.ConvTranspose2d(d_v, d_v, 3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(d_v, d_v // 2, 3, padding=1),
                nn.UpsamplingNearest2d(scale_factor=2)
            ),  # B x T x d_v/2 x H/4 x W/4
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 2, d_v // 2, 3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 2, d_v // 4, 3, padding=1),
                nn.UpsamplingNearest2d(scale_factor=2)
            ),  # B x T x d_v/4 x H/2 x W/2
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 4, d_v // 4, 3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 4, d_v // 8, 3, padding=1),
                nn.UpsamplingNearest2d(scale_factor=2)
            ),  # B x T x d_v/8 x H x W
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 8, d_v // 8, 3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 8, C, 3, padding=1),
                nn.Tanh()
            )  # B x T x C x H x W
        ])

        self.encoder = ConvTransformerEncoder(num_blocks, num_heads, d_v, d_ff)
        self.decoder = ConvTransformerDecoder(num_blocks, num_heads, d_v, d_ff)

    def forward_frame_encoder(self, input_frames):
        B, N, cur_C, cur_H, cur_W = input_frames.shape
        reps = [input_frames]

        for i, seq_layer in enumerate(self.frame_encoder):
            # Pass latest activations through the current frame encoder block
            last_rep_flat = reps[-1].contiguous().view(B * N, cur_C, cur_H, cur_W)
            seq_layer_output_flat = seq_layer(last_rep_flat)
            # Update record of activation sizes
            _, cur_C, cur_H, cur_W = seq_layer_output_flat.shape
            # Reshape and add to outputs
            seq_layer_output = seq_layer_output_flat.view(B, N, cur_C, cur_H, cur_W)
            reps.append(seq_layer_output)

        return reps[1:]

    def forward_frame_decoder(self, frame_decoder_inputs, frame_encoder_outputs):
        B, N, cur_C, cur_H, cur_W = frame_decoder_inputs.shape
        reps = [frame_decoder_inputs]

        for i, seq_layer in enumerate(self.frame_decoder):
            # Combine the decoder input and encoder output information
            comb_activations = reps[-1] + frame_encoder_outputs[-i-1]
            # Pass combined activations through the frame decoder block
            last_rep_flat = comb_activations.view(B * N, cur_C, cur_H, cur_W)
            seq_layer_output_flat = seq_layer(last_rep_flat)
            # Update record of activation sizes
            _, cur_C, cur_H, cur_W = seq_layer_output_flat.shape
            # Reshape and add to outputs
            seq_layer_output = seq_layer_output_flat.view(B, N, cur_C, cur_H, cur_W)
            reps.append(seq_layer_output)

        return reps[1:]

    @abstractmethod
    def forward(self, T, preceding_frames, following_frames):
        raise NotImplementedError()


class SCTSkipConScaledTForwardFillInModel(BaseSCTSkipConFillInModel):
    """Generates frames from earliest time step to latest."""

    def forward(self, T, preceding_frames, following_frames):
        """Forward method

        :param T: The number of new frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        :return: B x F x C x H x W
        """

        B, K, _, H, W = preceding_frames.shape
        F = following_frames.shape[1]
        use_cuda = module_is_cuda(self)

        # Create input mask
        encoder_input_mask = as_variable(torch.ones(B, K + F))
        if use_cuda:
            encoder_input_mask = encoder_input_mask.cuda()

        # Create input time steps [B x K+F]
        encoder_time_input = torch.cat([torch.arange(0, K) / K, torch.arange(0, F) / F + 2]) \
            .view(1, K + F).expand(B, K + F)
        encoder_time_input = as_variable(encoder_time_input)
        if use_cuda:
            encoder_time_input = encoder_time_input.cuda()

        # Combine preceding and following frame sequences
        input_frames = torch.cat([preceding_frames, following_frames], dim=1)  # B x K+F x C x H x W
        # Encode the input frames [B x K+F x d_v x H x W]
        encoder_input_reps = self.forward_frame_encoder(input_frames)
        encoder_output = self.encoder(encoder_input_reps[-1], encoder_input_mask, encoder_time_input)

        # Encode inputs to the decoder
        dec_input_frame_reps = self.forward_frame_encoder(preceding_frames[:, -1:, :, :, :])

        # Create decoder time steps [B x T]
        dec_time_input_full = (torch.arange(0, T) / T + 1).view(1, T).expand(B, T)
        dec_time_input_full = as_variable(dec_time_input_full)
        if use_cuda:
            dec_time_input_full = dec_time_input_full.cuda()

        # Create decoder product mask
        dec_prod_mask_full = torch.tril(torch.ones(T, T)).view(1, T, T).expand(B, T, T)
        dec_prod_mask_full = as_variable(dec_prod_mask_full)
        if use_cuda:
            dec_prod_mask_full = dec_prod_mask_full.cuda()

        # Pass information through the decoder
        decoder_output = self.decoder(encoder_output, encoder_input_mask, dec_input_frame_reps[-1], dec_time_input_full,
                                      dec_prod_mask_full)

        # Pass self-attention decoder outputs through image decoder
        output_reps = self.forward_frame_decoder(decoder_output, dec_input_frame_reps)
        pred_frames = output_reps[-1]

        return {
            'pred': pred_frames
        }


    def forward_train(self, preceding_frames, middle_frames, following_frames):
        """Forward method

        :param T: The number of new frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        :return: B x F x C x H x W
        """

        B, K, _, H, W = preceding_frames.shape
        T = middle_frames.shape[1]
        F = following_frames.shape[1]
        use_cuda = module_is_cuda(self)

        # Create input mask
        encoder_input_mask = as_variable(torch.ones(B, K + F))
        if use_cuda:
            encoder_input_mask = encoder_input_mask.cuda()

        # Create input time steps [B x K+F]
        encoder_time_input = torch.cat([torch.arange(0, K) / K, torch.arange(0, F) / F + 2]) \
            .view(1, K + F).expand(B, K + F)
        encoder_time_input = as_variable(encoder_time_input)
        if use_cuda:
            encoder_time_input = encoder_time_input.cuda()

        # Combine preceding and following frame sequences
        input_frames = torch.cat([preceding_frames, following_frames], dim=1)  # B x K+F x C x H x W
        # Encode the input frames [B x K+F x d_v x H x W]
        encoder_input_reps = self.forward_frame_encoder(input_frames)
        encoder_output = self.encoder(encoder_input_reps[-1], encoder_input_mask, encoder_time_input)

        # Encode inputs to the decoder
        if T > 1:
            dec_input_frames = torch.cat([preceding_frames[:, -1:, :, :, :], middle_frames[:, :-1, :, :, :]], dim=1)
        else:
            dec_input_frames = preceding_frames[:, -1:, :, :, :]
        dec_input_frame_reps = self.forward_frame_encoder(dec_input_frames)

        # Create decoder time steps [B x T]
        dec_time_input_full = (torch.arange(0, T) / T + 1).view(1, T).expand(B, T)
        dec_time_input_full = as_variable(dec_time_input_full)
        if use_cuda:
            dec_time_input_full = dec_time_input_full.cuda()

        # Create decoder product mask
        dec_prod_mask_full = torch.tril(torch.ones(T, T)).view(1, T, T).expand(B, T, T)
        dec_prod_mask_full = as_variable(dec_prod_mask_full)
        if use_cuda:
            dec_prod_mask_full = dec_prod_mask_full.cuda()

        # Pass information through the decoder
        decoder_output = self.decoder(encoder_output, encoder_input_mask, dec_input_frame_reps[-1], dec_time_input_full,
                                      dec_prod_mask_full)

        # Pass self-attention decoder outputs through image decoder
        output_reps = self.forward_frame_decoder(decoder_output, dec_input_frame_reps)
        pred_frames = output_reps[-1]

        return {
            'pred': pred_frames
        }


class SCTSkipConScaledTInwardFillInModel(BaseSCTSkipConFillInModel):
    """Generates the first frame, then the last frame, then the second-to-first, then the second-to-last, etc."""

    def forward(self, T, preceding_frames, following_frames):
        """Forward method

        :param T: The number of new frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        :return: B x F x C x H x W
        """

        B, K, _, H, W = preceding_frames.shape
        F = following_frames.shape[1]
        use_cuda = module_is_cuda(self)

        # Create input mask
        encoder_input_mask = as_variable(torch.ones(B, K + F))
        if use_cuda:
            encoder_input_mask = encoder_input_mask.cuda()

        # Create input time steps [B x K+F]
        encoder_time_input = torch.cat([torch.arange(0, K) / K, torch.arange(0, F) / F + 2]) \
            .view(1, K + F).expand(B, K + F)
        encoder_time_input = as_variable(encoder_time_input)
        if use_cuda:
            encoder_time_input = encoder_time_input.cuda()

        # Combine preceding and following frame sequences
        input_frames = torch.cat([preceding_frames, following_frames], dim=1)  # B x K+F x C x H x W
        # Encode the input frames [B x K+F x d_v x H x W
        encoder_input_reps = self.forward_frame_encoder(input_frames)
        encoder_output = self.encoder(encoder_input_reps[-1], encoder_input_mask, encoder_time_input)

        # Encode inputs to the decoder
        dec_input_frame_reps = self.forward_frame_encoder(preceding_frames[:, -1:, :, :, :])

        # Construct the order in which to generate frames
        a = torch.arange(0, T, step=0.5)
        b = torch.arange(T-0.5, -0.25, step=-0.5)
        mask = torch.remainder(torch.arange(2*T), 2)
        time_inputs = ((1-mask)*a + mask*b)[:T]
        # Create decoder time steps [B x T]
        dec_time_input_full = (time_inputs / T + 1).view(1, T).expand(B, T)
        dec_time_input_full = as_variable(dec_time_input_full)
        if use_cuda:
            dec_time_input_full = dec_time_input_full.cuda()

        # Create decoder product mask
        dec_prod_mask_full = torch.tril(torch.ones(T, T)).view(1, T, T).expand(B, T, T)
        dec_prod_mask_full = as_variable(dec_prod_mask_full)
        if use_cuda:
            dec_prod_mask_full = dec_prod_mask_full.cuda()

        # Pass information through the decoder
        decoder_output = self.decoder(encoder_output, encoder_input_mask, dec_input_frame_reps[-1], dec_time_input_full,
                                      dec_prod_mask_full)

        # Pass self-attention decoder outputs through image decoder
        output_reps = self.forward_frame_decoder(decoder_output, dec_input_frame_reps)

        # Extract and re-order frames
        pred_frames_permuted = output_reps[-1]
        _, order = torch.sort(dec_time_input_full)  # B x T
        order = order.view(B, T, 1, 1, 1).expand(B, T, self.C, H, W)
        pred_frames = torch.gather(pred_frames_permuted, 1, order)

        return {
            'pred': pred_frames
        }


class SCTSkipConScaledTRandomFillInModel(BaseSCTSkipConFillInModel):
    """Generates the middle frames in a random order."""

    def forward(self, T, preceding_frames, following_frames):
        """Forward method

        :param T: The number of new frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        :return: B x F x C x H x W
        """

        B, K, _, H, W = preceding_frames.shape
        F = following_frames.shape[1]
        use_cuda = module_is_cuda(self)

        # Create input mask
        encoder_input_mask = as_variable(torch.ones(B, K + F))
        if use_cuda:
            encoder_input_mask = encoder_input_mask.cuda()

        # Create input time steps [B x K+F]
        encoder_time_input = torch.cat([torch.arange(0, K) / K, torch.arange(0, F) / F + 2]) \
            .view(1, K + F).expand(B, K + F)
        encoder_time_input = as_variable(encoder_time_input)
        if use_cuda:
            encoder_time_input = encoder_time_input.cuda()

        # Combine preceding and following frame sequences
        input_frames = torch.cat([preceding_frames, following_frames], dim=1)  # B x K+F x C x H x W
        # Encode the input frames [B x K+F x d_v x H x W
        encoder_input_reps = self.forward_frame_encoder(input_frames)
        encoder_output = self.encoder(encoder_input_reps[-1], encoder_input_mask, encoder_time_input)

        # Encode inputs to the decoder
        dec_input_frame_reps = self.forward_frame_encoder(preceding_frames[:, -1:, :, :, :])

        # Randomly sample the order in which to generate middle frames
        middle_frame_indexes = [torch.randperm(T) for _ in xrange(B)]
        middle_frame_indexes = torch.stack(middle_frame_indexes)
        # Create decoder time steps [B x T]
        dec_time_input_full = (middle_frame_indexes.float() / T) + 1
        dec_time_input_full = as_variable(dec_time_input_full)
        if use_cuda:
            dec_time_input_full = dec_time_input_full.cuda()

        # Create decoder product mask
        dec_prod_mask_full = torch.tril(torch.ones(T, T)).view(1, T, T).expand(B, T, T)
        dec_prod_mask_full = as_variable(dec_prod_mask_full)
        if use_cuda:
            dec_prod_mask_full = dec_prod_mask_full.cuda()

        # Pass information through the decoder
        decoder_output = self.decoder(encoder_output, encoder_input_mask, dec_input_frame_reps[-1], dec_time_input_full,
                                      dec_prod_mask_full)

        # Pass self-attention decoder outputs through image decoder
        output_reps = self.forward_frame_decoder(decoder_output, dec_input_frame_reps)

        # Extract and reorder frames
        pred_frames_permuted = output_reps[-1]
        _, order = torch.sort(dec_time_input_full)  # B x T
        order = order.view(B, T, 1, 1, 1).expand(B, T, self.C, H, W)
        pred_frames = torch.gather(pred_frames_permuted, 1, order)

        return {
            'pred': pred_frames
        }


class SCTSkipConScaledTRandomBFillInModel(BaseSCTSkipConFillInModel):
    """Generates the middle frames in a random order. Passes the generated frames through the self-attention encoder,
    and only uses the self-attention decoder to produce the next frame."""

    INFTY = 1e8

    def forward(self, T, preceding_frames, following_frames):
        """Forward method

        :param T: The number of new frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        :return: B x F x C x H x W
        """

        B, K, _, H, W = preceding_frames.shape
        F = following_frames.shape[1]
        use_cuda = module_is_cuda(self)

        # Create time steps corresponding to preceding and following frames
        preceding_time_input = as_variable((torch.arange(0, K) / K).view(1, K).expand(B, K))
        following_time_input = as_variable((torch.arange(0, F) / F + 2).view(1, F).expand(B, F))
        if use_cuda:
            preceding_time_input = preceding_time_input.cuda()
            following_time_input = following_time_input.cuda()

        # Encode input frames
        enc_input_frame_reps_p = self.forward_frame_encoder(preceding_frames)  # B x K x d_v x H x W
        enc_input_frame_reps_f = self.forward_frame_encoder(following_frames)  # B x F x d_v x H x W

        # Initialize the mask of middle frame indexes that were generated
        chosen_frame_indexes_mask = as_variable(torch.zeros(B, T))
        if use_cuda:
            chosen_frame_indexes_mask = chosen_frame_indexes_mask.cuda()
        # Initialize lists for other values
        pred_frames_permuted = []
        enc_input_frame_top_reps_m = []
        middle_frame_indexes = []

        for t in xrange(T):
            if t == 0:
                encoder_input_top_rep = torch.cat([enc_input_frame_reps_p[-1], enc_input_frame_reps_f[-1]], dim=1)
                encoder_time_input = torch.cat([preceding_time_input, following_time_input], dim=1)
            else:
                # Frame-encode the last predicted frame
                last_pred_frame_reps = self.forward_frame_encoder(pred_frames_permuted[-1])
                # Update the list of top-level representations of the predicted frames
                enc_input_frame_top_reps_m.append(last_pred_frame_reps[-1])
                # Construct self-attention encoder input with top-level representations of predicted frames
                encoder_input_top_rep = torch.cat([enc_input_frame_reps_p[-1], enc_input_frame_reps_f[-1],
                                                   torch.cat(enc_input_frame_top_reps_m, dim=1)], dim=1)
                # Construct time inputs to self-attention encoder
                middle_frame_indexes_normalized = torch.cat(middle_frame_indexes, dim=1) / T + 1
                encoder_time_input = torch.cat([
                    preceding_time_input,
                    following_time_input,
                    middle_frame_indexes_normalized
                ], dim=1)

            # Pass generated frames through the self-attention encoder
            encoder_input_mask = as_variable(torch.ones(B, K + F + t))
            if use_cuda:
                encoder_input_mask = encoder_input_mask.cuda()
            encoder_output = self.encoder(encoder_input_top_rep, encoder_input_mask, encoder_time_input)

            # Select the index of the next frame to generate
            ones = as_variable((torch.ones(B, T) / T))
            if use_cuda:
                ones = ones.cuda()
            frame_index_prob_logits = ones * (-self.INFTY * chosen_frame_indexes_mask)  # B x T
            next_frame_indexes = torch.multinomial(nn.functional.softmax(frame_index_prob_logits, dim=1), 1)  # B x 1

            # Update the values associated with the frame indexes chosen so far
            chosen_frame_indexes_mask.scatter_(1, next_frame_indexes, 1)
            middle_frame_indexes.append(next_frame_indexes.float())

            # Get self-attention decoder input for current time step
            if t == 0:
                dec_input_frame_reps = self.forward_frame_encoder(preceding_frames[:, -1:, :, :, :])
            else:
                dec_input_frame_reps = last_pred_frame_reps

            # Construct decoder time input
            decoder_time_input = next_frame_indexes.float() / T + 1

            # Construct decoder product mask
            dec_prod_mask = as_variable(torch.ones(B, 1, 1))
            if use_cuda:
                dec_prod_mask = dec_prod_mask.cuda()

            # Pass information through the self-attention decoder
            decoder_output = self.decoder(encoder_output, encoder_input_mask, dec_input_frame_reps[-1],
                                          decoder_time_input, dec_prod_mask)

            # Pass self-attention decoder output through image encoder
            output_reps = self.forward_frame_decoder(decoder_output, dec_input_frame_reps)
            pred_frames_permuted.append(output_reps[-1])

        # Extract and reorder frames
        pred_frames_permuted = torch.cat(pred_frames_permuted, dim=1)
        _, order = torch.sort(torch.cat(middle_frame_indexes, dim=1))  # B x T
        order = order.view(B, T, 1, 1, 1).expand(B, T, self.C, H, W)
        pred_frames = torch.gather(pred_frames_permuted, 1, order)

        return {
            'pred': pred_frames
        }


class SCTSkipConScaledTRandomCFillInModel(BaseSCTSkipConFillInModel):
    """Generates the middle frames in a random order. Passes the generated frames through the self-attention encoder,
    and only uses the self-attention decoder to produce the next frame. Employs skip connections between the
    self-attention encoder and decoder."""

    INFTY = 1e8

    def __init__(self, C, num_blocks, num_heads, d_v, d_ff):
        super(SCTSkipConScaledTRandomCFillInModel, self).__init__(C, num_blocks, num_heads, d_v, d_ff)

        self.encoder = ConvSkipConTransformerEncoder(num_blocks, num_heads, d_v, d_ff)
        self.decoder = ConvSkipConTransformerDecoder(num_blocks, num_heads, d_v, d_ff)

    def forward(self, T, preceding_frames, following_frames):
        """Forward method

        :param T: The number of new frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        :return: B x F x C x H x W
        """

        B, K, _, H, W = preceding_frames.shape
        F = following_frames.shape[1]
        use_cuda = module_is_cuda(self)

        # Create time steps corresponding to preceding and following frames
        preceding_time_input = as_variable((torch.arange(0, K) / K).view(1, K).expand(B, K))
        following_time_input = as_variable((torch.arange(0, F) / F + 2).view(1, F).expand(B, F))
        if use_cuda:
            preceding_time_input = preceding_time_input.cuda()
            following_time_input = following_time_input.cuda()

        # Encode input frames
        enc_input_frame_reps_p = self.forward_frame_encoder(preceding_frames)  # B x K x d_v x H x W
        enc_input_frame_reps_f = self.forward_frame_encoder(following_frames)  # B x F x d_v x H x W

        # Initialize the mask of middle frame indexes that were generated
        chosen_frame_indexes_mask = as_variable(torch.zeros(B, T))
        if use_cuda:
            chosen_frame_indexes_mask = chosen_frame_indexes_mask.cuda()
        # Initialize lists for other values
        pred_frames_permuted = []
        enc_input_frame_top_reps_m = []
        middle_frame_indexes = []

        for t in xrange(T):
            if t == 0:
                encoder_input_top_rep = torch.cat([enc_input_frame_reps_p[-1], enc_input_frame_reps_f[-1]], dim=1)
                encoder_time_input = torch.cat([preceding_time_input, following_time_input], dim=1)
            else:
                # Frame-encode the last predicted frame
                last_pred_frame_reps = self.forward_frame_encoder(pred_frames_permuted[-1])
                # Update the list of top-level representations of the predicted frames
                enc_input_frame_top_reps_m.append(last_pred_frame_reps[-1])
                # Construct self-attention encoder input with top-level representations of predicted frames
                encoder_input_top_rep = torch.cat([enc_input_frame_reps_p[-1], enc_input_frame_reps_f[-1],
                                                   torch.cat(enc_input_frame_top_reps_m, dim=1)], dim=1)
                # Construct time inputs to self-attention encoder
                middle_frame_indexes_normalized = torch.cat(middle_frame_indexes, dim=1) / T + 1
                encoder_time_input = torch.cat([
                    preceding_time_input,
                    following_time_input,
                    middle_frame_indexes_normalized
                ], dim=1)

            # Pass generated frames through the self-attention encoder
            encoder_input_mask = as_variable(torch.ones(B, K + F + t))
            if use_cuda:
                encoder_input_mask = encoder_input_mask.cuda()
            encoder_outputs = self.encoder(encoder_input_top_rep, encoder_input_mask, encoder_time_input)

            # Select the index of the next frame to generate
            ones = as_variable((torch.ones(B, T) / T))
            if use_cuda:
                ones = ones.cuda()
            frame_index_prob_logits = ones * (-self.INFTY * chosen_frame_indexes_mask)  # B x T
            next_frame_indexes = torch.multinomial(nn.functional.softmax(frame_index_prob_logits, dim=1), 1)  # B x 1

            # Update the values associated with the frame indexes chosen so far
            chosen_frame_indexes_mask.scatter_(1, next_frame_indexes, 1)
            middle_frame_indexes.append(next_frame_indexes.float())

            # Get self-attention decoder input for current time step
            if t == 0:
                dec_input_frame_reps = self.forward_frame_encoder(preceding_frames[:, -1:, :, :, :])
            else:
                dec_input_frame_reps = last_pred_frame_reps

            # Construct decoder time input
            decoder_time_input = next_frame_indexes.float() / T + 1

            # Construct decoder product mask
            dec_prod_mask = as_variable(torch.ones(B, 1, 1))
            if use_cuda:
                dec_prod_mask = dec_prod_mask.cuda()

            # Pass information through the self-attention decoder
            decoder_output = self.decoder(encoder_outputs, encoder_input_mask, dec_input_frame_reps[-1],
                                          decoder_time_input, dec_prod_mask)

            # Pass self-attention decoder output through image encoder
            output_reps = self.forward_frame_decoder(decoder_output, dec_input_frame_reps)
            pred_frames_permuted.append(output_reps[-1])

        # Extract and reorder frames
        pred_frames_permuted = torch.cat(pred_frames_permuted, dim=1)
        _, order = torch.sort(torch.cat(middle_frame_indexes, dim=1))  # B x T
        order = order.view(B, T, 1, 1, 1).expand(B, T, self.C, H, W)
        pred_frames = torch.gather(pred_frames_permuted, 1, order)

        return {
            'pred': pred_frames
        }


class SCTSkipConScaledTRandomDFillInModel(SCTSkipConScaledTRandomCFillInModel):
    """Generates the middle frames in a random order. Passes the generated frames through the self-attention encoder,
    and only uses the self-attention decoder to produce the next frame. The decoder attends to the corresponding
    level in the encoder."""

    def __init__(self, C, num_blocks, num_heads, d_v, d_ff):
        super(SCTSkipConScaledTRandomCFillInModel, self).__init__(C, num_blocks, num_heads, d_v, d_ff)

        self.encoder = ConvSkipConTransformerEncoder(num_blocks, num_heads, d_v, d_ff)
        self.decoder = ConvHiddenTransformerDecoder(num_blocks, num_heads, d_v, d_ff)


class SCTBypassScaledTForwardFillInModel(SCTSkipConScaledTForwardFillInModel):
    """Generates middle frames from earliest time step to latest. Instead of a skip connection, the encoded input
    frames are passed through more convolutional layers."""

    def __init__(self, C, num_blocks, num_heads, d_v, d_ff):
        super(SCTBypassScaledTForwardFillInModel, self).__init__(C, num_blocks, num_heads, d_v, d_ff)

        bypass_layers = []
        for i in xrange(4):
            num_feats = d_v // 2 ** (3-i)
            bypass_layers.append(nn.Sequential(
                nn.Conv2d(num_feats, num_feats, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(num_feats, num_feats, 3, padding=1),
                nn.ReLU()
            ))
        self.bypass_layers = nn.ModuleList(bypass_layers)


    def forward_bypass_layer(self, bypass_layer, frame_encoder_outputs):
        B, T, d, H, W = frame_encoder_outputs.shape
        frame_encoder_outputs_flat = frame_encoder_outputs.contiguous().view(B * T, d, H, W)
        bypass_output_flat = bypass_layer(frame_encoder_outputs_flat)
        bypass_output = bypass_output_flat.view(B, T, d, H, W)

        return bypass_output


    def forward_frame_decoder(self, frame_decoder_inputs, frame_encoder_outputs):
        B, N, cur_C, cur_H, cur_W = frame_decoder_inputs.shape
        reps = [frame_decoder_inputs]

        for i, seq_layer in enumerate(self.frame_decoder):
            # Combine the decoder input and encoder output information
            comb_activations = reps[-1] + self.forward_bypass_layer(self.bypass_layers[-i-1],
                                                                    frame_encoder_outputs[-i-1])
            # Pass combined activations through the frame decoder block
            last_rep_flat = comb_activations.view(B * N, cur_C, cur_H, cur_W)
            seq_layer_output_flat = seq_layer(last_rep_flat)
            # Update record of activation sizes
            _, cur_C, cur_H, cur_W = seq_layer_output_flat.shape
            # Reshape and add to outputs
            seq_layer_output = seq_layer_output_flat.view(B, N, cur_C, cur_H, cur_W)
            reps.append(seq_layer_output)

        return reps[1:]


class SCTFrameEncDecBNSkipConScaledTForwardFillInModel(SCTSkipConScaledTForwardFillInModel):

    def __init__(self, C, num_blocks, num_heads, d_v, d_ff):
        """Constructor

        :param num_blocks: Number of blocks in the encoder stack
        :param num_heads: Number of heads to use for each multi-head attention module
        :param d_v: Dimensionality (number of features) of the values
        :param d_ff: Intermediate dimensionality of the "point-wise" feed-forward layers
        :param C: Number of input channels
        """

        super(BaseSCTSkipConFillInModel, self).__init__()

        self.d_v = d_v
        self.C = C

        self.frame_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C, d_v // 8, 3, padding=1),
                nn.BatchNorm2d(d_v // 8),
                nn.ReLU(),
                nn.Conv2d(d_v // 8, d_v // 8, 3, padding=1),
                nn.BatchNorm2d(d_v // 8)
            ),  # B x T x d_v/8 x H x W
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(d_v // 8, d_v // 4, 3, padding=1),
                nn.BatchNorm2d(d_v // 4),
                nn.ReLU(),
                nn.Conv2d(d_v // 4, d_v // 4, 3, padding=1),
                nn.BatchNorm2d(d_v // 4)
            ),  # B x T x d_v/4 x H/2 x W/2
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(d_v // 4, d_v // 2, 3, padding=1),
                nn.BatchNorm2d(d_v // 2),
                nn.ReLU(),
                nn.Conv2d(d_v // 2, d_v // 2, 3, padding=1),
                nn.BatchNorm2d(d_v // 2)
            ),  # B x T x d_v/2 x H/4 x W/4
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(d_v // 2, d_v, 3, padding=1),
                nn.BatchNorm2d(d_v),
                nn.ReLU(),
                nn.Conv2d(d_v, d_v, 3, padding=1),
                nn.BatchNorm2d(d_v)
            )  # B x T x d_v x H/8 x W/8
        ])

        self.frame_decoder = nn.ModuleList([
            # B x T x d_v x H/8 x W/8
            nn.Sequential(
                nn.ConvTranspose2d(d_v, d_v, 3, padding=1),
                nn.BatchNorm2d(d_v),
                nn.ReLU(),
                nn.ConvTranspose2d(d_v, d_v // 2, 3, padding=1),
                nn.BatchNorm2d(d_v // 2),
                nn.UpsamplingNearest2d(scale_factor=2)
            ),  # B x T x d_v/2 x H/4 x W/4
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 2, d_v // 2, 3, padding=1),
                nn.BatchNorm2d(d_v // 2),
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 2, d_v // 4, 3, padding=1),
                nn.BatchNorm2d(d_v // 4),
                nn.UpsamplingNearest2d(scale_factor=2)
            ),  # B x T x d_v/4 x H/2 x W/2
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 4, d_v // 4, 3, padding=1),
                nn.BatchNorm2d(d_v // 4),
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 4, d_v // 8, 3, padding=1),
                nn.BatchNorm2d(d_v // 8),
                nn.UpsamplingNearest2d(scale_factor=2)
            ),  # B x T x d_v/8 x H x W
            nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 8, d_v // 8, 3, padding=1),
                nn.BatchNorm2d(d_v // 8),
                nn.ReLU(),
                nn.ConvTranspose2d(d_v // 8, C, 3, padding=1),
                nn.Tanh()
            )  # B x T x C x H x W
        ])

        self.encoder = ConvTransformerEncoder(num_blocks, num_heads, d_v, d_ff)
        self.decoder = ConvTransformerDecoder(num_blocks, num_heads, d_v, d_ff)