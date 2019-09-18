import torch
from torch import nn as nn

from src.models.self_attention.submodules import ConvTransformerEncoder, ConvTransformerDecoder, \
    ConvSkipConTransformerEncoder, ConvSkipConTransformerDecoder
from src.util.util import module_is_cuda, as_variable


class SCTSkipConFillInModel(nn.Module):

    def __init__(self, C, num_blocks, num_heads, d_v, d_ff):
        """Constructor

        :param num_blocks: Number of blocks in the encoder stack
        :param num_heads: Number of heads to use for each multi-head attention module
        :param d_v: Dimensionality (number of features) of the values
        :param d_ff: Intermediate dimensionality of the "point-wise" feed-forward layers
        :param C: Number of input channels
        """

        super(SCTSkipConFillInModel, self).__init__()

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
        encoder_time_input = torch.cat([torch.arange(0, K), torch.arange(K + T, K + T + F)]) \
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

        # Create decoder time steps [B x T]
        dec_time_input_full = torch.arange(K, K + T).view(1, T).expand(B, T)
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

        return {
            'pred': output_reps[-1]
        }


    def forward_train(self, preceding_frames, middle_frames, following_frames):
        """Forward method used during training. This has access to the middle frames, so it can do a single forward
        pass to compute all next frames.

        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param middle_frames: The frames to predict (B x T x C x H x W)
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
        encoder_time_input = torch.cat([torch.arange(0, K), torch.arange(K + T, K + T + F)]).view(1, K + F).expand(B, K + F)
        encoder_time_input = as_variable(encoder_time_input)
        if use_cuda:
            encoder_time_input = encoder_time_input.cuda()

        # Combine preceding and following frame sequences
        input_frames = torch.cat([preceding_frames, following_frames], dim=1)  # B x K+F x C x H x W
        # Encode the input frames
        encoder_input_reps = self.forward_frame_encoder(input_frames)
        encoder_input = encoder_input_reps[-1]
        encoder_output = self.encoder(encoder_input, encoder_input_mask, encoder_time_input)  # B x K+F x d_v x H x W

        # Encode inputs to the decoder. Skip the last middle frame
        frame_encoder_input = torch.cat([preceding_frames[:, -1:, :, :, :], middle_frames[:, :-1, :, :, :]], dim=1)
        dec_input_frame_reps = self.forward_frame_encoder(frame_encoder_input)

        # Create decoder time steps [B x T]
        dec_time_input = torch.arange(K, K + T).view(1, T).expand(B, T)
        dec_time_input = as_variable(dec_time_input)
        if use_cuda:
            dec_time_input = dec_time_input.cuda()

        # Create decoder product mask
        dec_prod_mask = torch.tril(torch.ones(T, T)).view(1, T, T).expand(B, T, T)
        dec_prod_mask = as_variable(dec_prod_mask)
        if use_cuda:
            dec_prod_mask = dec_prod_mask.cuda()

        # Pass information through the decoder
        decoder_output = self.decoder(encoder_output, encoder_input_mask, dec_input_frame_reps[-1], dec_time_input,
                                      dec_prod_mask)
        # Pass self-attention decoder outputs through image decoder
        output_reps = self.forward_frame_decoder(decoder_output, dec_input_frame_reps)

        return {
            'pred': output_reps[-1]
        }


class SCTSuperSkipConFillInModel(nn.Module):

    def __init__(self, C, num_blocks, num_heads, d_v, d_ff):
        """Constructor

        :param num_blocks: Number of blocks in the encoder stack
        :param num_heads: Number of heads to use for each multi-head attention module
        :param d_v: Dimensionality (number of features) of the values
        :param d_ff: Intermediate dimensionality of the "point-wise" feed-forward layers
        :param C: Number of input channels
        """

        super(SCTSuperSkipConFillInModel, self).__init__()

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

        self.encoder = ConvSkipConTransformerEncoder(num_blocks, num_heads, d_v, d_ff)
        self.decoder = ConvSkipConTransformerDecoder(num_blocks, num_heads, d_v, d_ff)


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
        encoder_time_input = torch.cat([torch.arange(0, K), torch.arange(K + T, K + T + F)]) \
            .view(1, K + F).expand(B, K + F)
        encoder_time_input = as_variable(encoder_time_input)
        if use_cuda:
            encoder_time_input = encoder_time_input.cuda()

        # Combine preceding and following frame sequences
        input_frames = torch.cat([preceding_frames, following_frames], dim=1)  # B x K+F x C x H x W
        # Encode the input frames [B x K+F x d_v x H x W
        encoder_input_reps = self.forward_frame_encoder(input_frames)
        encoder_outputs = self.encoder(encoder_input_reps[-1], encoder_input_mask, encoder_time_input)

        # Create start token
        start_token = torch.zeros(B, 1, self.C, H, W)
        start_token = as_variable(start_token)
        if use_cuda:
            start_token = start_token.cuda()

        # Encode inputs to the decoder
        dec_input_frame_reps = self.forward_frame_encoder(start_token)

        # Create decoder time steps [B x T]
        dec_time_input_full = torch.arange(K, K + T).view(1, T).expand(B, T)
        dec_time_input_full = as_variable(dec_time_input_full)
        if use_cuda:
            dec_time_input_full = dec_time_input_full.cuda()

        # Create decoder product mask
        dec_prod_mask_full = torch.tril(torch.ones(T, T)).view(1, T, T).expand(B, T, T)
        dec_prod_mask_full = as_variable(dec_prod_mask_full)
        if use_cuda:
            dec_prod_mask_full = dec_prod_mask_full.cuda()

        # Pass information through the decoder
        decoder_output = self.decoder(encoder_outputs, encoder_input_mask, dec_input_frame_reps[-1],
                                      dec_time_input_full, dec_prod_mask_full)

        # Pass self-attention decoder outputs through image decoder
        output_reps = self.forward_frame_decoder(decoder_output, dec_input_frame_reps)

        return {
            'pred': output_reps[-1]
        }


    def forward_train(self, preceding_frames, middle_frames, following_frames):
        """Forward method used during training. This has access to the middle frames, so it can do a single forward
        pass to compute all next frames.

        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param middle_frames: The frames to predict (B x T x C x H x W)
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
        encoder_time_input = torch.cat([torch.arange(0, K), torch.arange(K + T, K + T + F)]).view(1, K + F).expand(B, K + F)
        encoder_time_input = as_variable(encoder_time_input)
        if use_cuda:
            encoder_time_input = encoder_time_input.cuda()

        # Combine preceding and following frame sequences
        input_frames = torch.cat([preceding_frames, following_frames], dim=1)  # B x K+F x C x H x W
        # Encode the input frames
        encoder_input_reps = self.forward_frame_encoder(input_frames)
        encoder_input = encoder_input_reps[-1]
        encoder_outputs = self.encoder(encoder_input, encoder_input_mask, encoder_time_input)  # B x K+F x d_v x H x W

        # Create start token
        start_token = torch.zeros(B, 1, self.C, H, W)
        start_token = as_variable(start_token)
        if use_cuda:
            start_token = start_token.cuda()

        # Encode inputs to the decoder. Skip the last middle frame
        frame_encoder_input = torch.cat([start_token, middle_frames[:, :-1, :, :, :]], dim=1)
        dec_input_frame_reps = self.forward_frame_encoder(frame_encoder_input)

        # Create decoder time steps [B x T]
        dec_time_input = torch.arange(K, K + T).view(1, T).expand(B, T)
        dec_time_input = as_variable(dec_time_input)
        if use_cuda:
            dec_time_input = dec_time_input.cuda()

        # Create decoder product mask
        dec_prod_mask = torch.tril(torch.ones(T, T)).view(1, T, T).expand(B, T, T)
        dec_prod_mask = as_variable(dec_prod_mask)
        if use_cuda:
            dec_prod_mask = dec_prod_mask.cuda()

        # Pass information through the decoder
        decoder_output = self.decoder(encoder_outputs, encoder_input_mask, dec_input_frame_reps[-1], dec_time_input,
                                      dec_prod_mask)
        # Pass self-attention decoder outputs through image decoder
        output_reps = self.forward_frame_decoder(decoder_output, dec_input_frame_reps)

        return {
            'pred': output_reps[-1]
        }