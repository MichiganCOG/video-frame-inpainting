import math

import torch
from torch import nn as nn

from src.util.util import module_is_cuda, as_variable
from ...discriminators.SNDiscriminator import SNConv2d, SNLinear


class ScaledDotProductAttention(nn.Module):

    def __init__(self, inf=1e10):
        """Constructor

        :param inf: The value to use for infinity
        """

        super(ScaledDotProductAttention, self).__init__()

        self.inf = inf

        self.softmax = nn.Softmax(dim=2)


    def forward(self, q, k, v, prod_mask=None):
        """Forward method

        :param q: Query tensor [B x T_o x d_qk FloatTensor Variable]
        :param k: Key tensor [B x T_i x d_qk FloatTensor Variable]
        :param v: Value tensor [B x T_i x ... FloatTensor]
        :param prod_mask: Binary mask of values that can be attended to [B x T_o x T_i ByteTensor]. If entry
                          (b, i, j) is 0, then the i'th output cannot depend on the j'th input for the b'th item.
        :return: A sequence of attended values [B x T_o x ...]
        """

        B = v.size(0)
        T_i = v.size(1)
        T_o = q.size(1)
        d_qk = q.size(2)

        # Compute weight of each value across the sequence of queries
        qk = torch.matmul(q, k.transpose(1, 2))  # B x T_o x T_i
        scaled_qk = qk / math.sqrt(d_qk)
        if prod_mask is not None:
            scaled_qk.masked_fill_(prod_mask == 0, -self.inf)
        weights = self.softmax(scaled_qk)  # B x T_o x T_i

        # Compute weighted sum over flattened values
        v_flat = v.contiguous().view(B, T_i, -1)
        attended_v = torch.matmul(weights, v_flat)
        # Reshape values to original dimensions
        attended_v = attended_v.view(B, T_o, *v.shape[2:])  # B x T_o x ...

        return attended_v


class ConvMultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_qk, d_v, use_sn=False):
        """Constructor

        :param num_heads: Number of heads to use
        :param d_qk: Dimensionality of the queries and keys
        :param d_v: Dimensionality (number of features) of the values
        """

        super(ConvMultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_v = d_v

        linear = SNLinear if use_sn else nn.Linear
        conv2d = SNConv2d if use_sn else nn.Conv2d

        self.q_layer = linear(d_qk, num_heads * d_qk)
        self.k_layer = linear(d_qk, num_heads * d_qk)
        self.v_layer = conv2d(d_v, num_heads * d_v, 3, padding=1)

        self.attn_module = ScaledDotProductAttention()

        self.proj_concat_layer = nn.Conv2d(num_heads * d_v, d_v, 1)


    def forward(self, v, k, q, prod_mask=None):
        """Forward method

        :param q: Query tensor [B x T_o x d_qk FloatTensor Variable]
        :param k: Key tensor [B x T_i x d_qk FloatTensor Variable]
        :param v: Value tensor [B x T_i x d_v x H x W FloatTensor]
        :param prod_mask: Binary mask of values that can be attended to [B x T_o x T_i ByteTensor]. If entry
                          (b, i, j) is 0, then the i'th output cannot depend on the j'th input for the b'th item.
        :return: A sequence of (combined and transformed) attended values [B x T_o x d_v x H x W]
        """

        B, T_i, _, H, W = v.shape
        T_o = q.size(1)

        # Combine the batch and time dimensions of the given values
        v_combined_B_T_i = v.contiguous().view(-1, self.d_v, H, W)

        # Project queries, keys, values
        proj_q = self.q_layer(q)
        proj_k = self.k_layer(k)
        proj_v_combined_B_T_i = self.v_layer(v_combined_B_T_i)
        # Restore dimensions of projected v
        proj_v = proj_v_combined_B_T_i.view(B, T_i, self.num_heads * self.d_v, H, W)

        # Split into inputs to each attention module
        proj_qs = torch.chunk(proj_q, self.num_heads, dim=2)
        proj_ks = torch.chunk(proj_k, self.num_heads, dim=2)
        proj_vs = torch.chunk(proj_v, self.num_heads, dim=2)

        # Pass each chunk through the attention module
        concat_inputs = []
        for h in xrange(self.num_heads):
            attn_module_output = self.attn_module(proj_qs[h], proj_ks[h], proj_vs[h], prod_mask=prod_mask)
            concat_inputs.append(attn_module_output)

        # Concatenate
        concat = torch.cat(concat_inputs, dim=2)  # B x T_o x num_heads*d_v x H x W
        # Project the attention module outputs (need to combine B and T_o dims first, then project, then reshape)
        concat_combined_B_T_o = concat.view(-1, self.num_heads * self.d_v, H, W)
        proj_concat_combined_B_T_o = self.proj_concat_layer(concat_combined_B_T_o)
        proj_concat = proj_concat_combined_B_T_o.view(B, T_o, self.d_v, H, W)

        return proj_concat


class ConvTransformerEncoderBlock(nn.Module):

    def __init__(self, num_heads, d_v, d_ff, use_sn=False):
        """Constructor

        :param num_heads: The number of heads to use in multi-head attention
        :param d_v: Dimensionality (number of features) of the values
        :param d_ff: Intermediate dimensionality of the feed-forward part of this encoder block
        """

        super(ConvTransformerEncoderBlock, self).__init__()

        self.d_v = d_v

        conv2d = SNConv2d if use_sn else nn.Conv2d

        self.batch_norm = nn.BatchNorm3d(d_v)
        self.mha_module = ConvMultiHeadAttention(num_heads, d_v, d_v, use_sn=use_sn)
        self.ff_module = nn.Sequential(
            conv2d(d_v, d_ff, 3, padding=1),
            nn.ReLU(inplace=True),
            conv2d(d_ff, d_v, 3, padding=1)
        )


    def forward(self, v, seq_mask=None):
        """Forward method

        :param v: Value tensor [B x T_i x d_v x H x W FloatTensor]
        :param seq_mask: The mask for items in the sequence that can be attended to [B x T_i ByteTensor]
        :return: B x T_i x d_v x H x W
        """

        B, T_i, _, H, W = v.shape
        use_cuda = module_is_cuda(self)

        # Prepare full mask over inputs
        if seq_mask is None:
            seq_mask = torch.ones(B, T_i)
            if use_cuda:
                seq_mask = seq_mask.cuda()
            seq_mask = as_variable(seq_mask)

        # Compute masked v
        expanded_seq_mask = seq_mask.contiguous().view(B, T_i, 1, 1, 1).expand(B, T_i, self.d_v, H, W)
        masked_v = expanded_seq_mask * v

        # Compute keys and queries
        q = masked_v.mean(-1).mean(-1)
        # Pass block input through multi-head attention (note: masked values will ignored due to prod_mask)
        prod_mask = seq_mask.view(B, 1, T_i).expand(B, T_i, T_i)
        mha_output = self.mha_module(v, q, q, prod_mask=prod_mask)  # B x T_i x d_v x H x W
        # Apply skip connection and normalization
        ff_input = self.apply_batch_norm(masked_v + mha_output)

        # Pass output through feed-forward module
        ff_input_combined_B_T_i = ff_input.contiguous().view(-1, self.d_v, H, W)  # B*T_i x d_v x H x W
        ff_output_combined_B_T_i = self.ff_module(ff_input_combined_B_T_i)  # B*T_i x d_v x H x W
        ff_output = ff_output_combined_B_T_i.view(B, T_i, self.d_v, H, W)  # B x T_i x d_v x H x W
        # Apply skip connection and layer normalization
        block_output = self.apply_batch_norm(ff_input + ff_output)

        return block_output


    def apply_batch_norm(self, v):
        """Applies batch normalization to the given value tensor. Swaps the time and feature dimensions for
        compatibility with BatchNorm3d.

        :param v: A value tensor [B x T x d_v x H x W]
        :return: Normalized value tensor [B x T x d_v x H x W]
        """
        v_permuted = v.permute(0, 2, 1, 3, 4).contiguous()
        output_permuted = self.batch_norm(v_permuted)
        output = output_permuted.permute(0, 2, 1, 3, 4)

        return output


class ConvTransformerDecoderBlock(nn.Module):

    def __init__(self, num_heads, d_v, d_ff):
        """Constructor

        :param num_heads: The number of heads to use in multi-head attention
        :param d_v: Dimensionality (number of features) of the values
        :param d_ff: Intermediate dimensionality of the feed-forward part of this decoder block
        """
        super(ConvTransformerDecoderBlock, self).__init__()

        self.d_v = d_v

        self.batch_norm = nn.BatchNorm3d(d_v)
        # Multi-head attention module that operates on the decoder input
        self.dec_only_mha_module = ConvMultiHeadAttention(num_heads, d_v, d_v)
        # Multi-head attention module that operates on the encoder output and (transformed) decoder input
        self.comb_enc_dec_mha_module = ConvMultiHeadAttention(num_heads, d_v, d_v)
        # Feed-forward module
        self.ff_module = nn.Sequential(
            nn.Conv2d(d_v, d_ff, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_ff, d_v, 3, padding=1)
        )


    def forward(self, q_dec, kv_dec, kv_enc, enc_seq_mask=None, prod_mask=None):
        """Forward method

        Note: q_dec and kv_dec are separate because you sometimes have to decode a different number of steps from
        what you computed so far. A concrete example is decoding one step at a time (you have to attend to all
        previously-generated outputs [T_o_old], but you're only producing one output at a time [T_o_new]).

        :param q_dec: The term used to compute Q in the decoder input MHA module [B x T_o_new x d_v x H x W]
        :param kv_dec: The term used for both K and V in the decoder input MHA module [B x T_o_old x d_v x H x W]
        :param kv_enc: The term used for both K and V in the combined encoder-decoder MHA module [B x T_i x d_v x H x W]
        :param enc_seq_mask: The mask for items in the encoder output that can be attended to [B x T_i ByteTensor]
        :param prod_mask: Binary mask of values that can be attended to in the decoder input
                          [B x T_o_new x T_o_old ByteTensor]. If entry (b, i, j) is 0, then the i'th output cannot
                          depend on the j'th decoder input for the b'th item.
        :return: B x T_o_new x d_v x H x W
        """

        B, T_o_new, _, H, W = q_dec.shape
        _, T_i, _, _, _ = kv_enc.shape
        use_cuda = module_is_cuda(self)

        # Prepare sequence mask if none is provided
        if enc_seq_mask is None:
            enc_seq_mask = torch.ones(B, T_i)
            if use_cuda:
                enc_seq_mask = enc_seq_mask.cuda()
            enc_seq_mask = as_variable(enc_seq_mask)
        # Expand encoding sequence mask into a product mask
        enc_prod_mask = enc_seq_mask.view(B, 1, T_i).expand(B, T_o_new, T_i)

        # Apply MHA module on decoder inputs
        q_dec_vec = q_dec.mean(-1).mean(-1)  # B x T_o_new x d_v
        kv_dec_vec = kv_dec.mean(-1).mean(-1)  # B x T_o_old x d_v
        # B x T_o_new x d_v x H x W
        dec_only_mha_output = self.dec_only_mha_module.forward(kv_dec, kv_dec_vec, q_dec_vec, prod_mask=prod_mask)
        # Apply skip connection and normalization
        comb_enc_dec_mha_input = self.apply_batch_norm(dec_only_mha_output + q_dec)  # B x T_o_new x d_v x H x W

        # Apply MHA module to combine decoder and encoder information
        kv_enc_vec = kv_enc.mean(-1).mean(-1)  # B x T_i x d_v
        comb_enc_dec_mha_input_vec = comb_enc_dec_mha_input.mean(-1).mean(-1)  # B x T_o_new x d_v
        comb_enc_dec_mha_output = self.comb_enc_dec_mha_module(kv_enc, kv_enc_vec, comb_enc_dec_mha_input_vec,
                                                               prod_mask=enc_prod_mask)
        ff_input = self.apply_batch_norm(comb_enc_dec_mha_output + comb_enc_dec_mha_input)

        # Pass output through feed-forward module
        ff_input_combined_B_T_i = ff_input.contiguous().view(-1, self.d_v, H, W)  # B*T_o_new x d_v x H x W
        ff_output_combined_B_T_i = self.ff_module(ff_input_combined_B_T_i)  # B*T_o_new x d_v x H x W
        ff_output = ff_output_combined_B_T_i.view(B, T_o_new, self.d_v, H, W)  # B x T_o_new x d_v x H x W
        # Apply skip connection and layer normalization
        block_output = self.apply_batch_norm(ff_input + ff_output)

        return block_output


    def apply_batch_norm(self, v):
        """Applies batch normalization to the given value tensor. Swaps the time and feature dimensions for
        compatibility with BatchNorm3d.

        :param v: A value tensor [B x T x d_v x H x W]
        :return: Normalized value tensor [B x T x d_v x H x W]
        """
        v_permuted = v.permute(0, 2, 1, 3, 4).contiguous()
        output_permuted = self.batch_norm(v_permuted)
        output = output_permuted.permute(0, 2, 1, 3, 4)

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_v):
        """Constructor

        :param d_v: Dimensionality of the positional encoding
        """
        super(PositionalEncoding, self).__init__()

        self.d_v = d_v

        self.denoms = as_variable(torch.pow(10000, 2 * torch.arange(0, d_v/2) / d_v))


    def forward(self, t):
        """Forward method

        :param t: A LongTensor containing the time steps to transform [B x T]
        :return: B x T x d_v
        """
        use_cuda = t.is_cuda

        B, T = t.shape
        pos = t.contiguous().view(B, T, 1).expand(B, T, self.d_v / 2)  # B x T x d_v/2
        denoms = self.denoms if not use_cuda else self.denoms.cuda()
        wave_inputs = pos / denoms.view(1, 1, self.d_v / 2)  # B x T x d_v/2

        pe_sin = torch.sin(wave_inputs) # B x T x d_v/2
        pe_cos = torch.cos(wave_inputs) # B x T x d_v/2

        pe = torch.stack([pe_sin, pe_cos], dim=-1)  # B x T x d_v/2 x 2
        pe = pe.view(B, T, self.d_v)  # B x T x d_v

        return pe


class SpatialPositionalEncodingAdder(nn.Module):

    def __init__(self, d_v):
        """Constructor

        :param d_v: Dimensionality of the positional encoding
        """
        super(SpatialPositionalEncodingAdder, self).__init__()

        self.d_v = d_v

        self.pos_enc_layer = PositionalEncoding(d_v)


    def forward(self, input, t):
        """Forward method

        :param input: The input to modulate with time [B x T x d_v x H x W]
        :param t: A LongTensor containing the time steps to transform [B x T]
        :return: B x T x d_v x H x W
        """
        B, T, _, H, W = input.shape

        pe = self.pos_enc_layer(t)
        pe_tiled = pe.view(B, T, self.d_v, 1, 1).expand(B, T, self.d_v, H, W)

        return input + pe_tiled


class ConvTransformerEncoder(nn.Module):

    def __init__(self, num_blocks, num_heads, d_v, d_ff, use_sn=False):
        """Constructor

        :param num_blocks: Number of blocks in the encoder stack
        :param num_heads: Number of heads to use for each multi-head attention module
        :param d_v: Dimensionality (number of features) of the values
        :param d_ff: Intermediate dimensionality of the "point-wise" feed-forward layers
        """

        super(ConvTransformerEncoder, self).__init__()

        self.num_blocks = num_blocks

        self.spatial_pos_enc_adder = SpatialPositionalEncodingAdder(d_v)

        self.encoder_blocks = nn.ModuleList()
        for _ in xrange(num_blocks):
            self.encoder_blocks.append(ConvTransformerEncoderBlock(num_heads, d_v=d_v, d_ff=d_v, use_sn=use_sn))


    def forward(self, input, input_mask, time_steps):
        """Forward method

        :param input: The standard input to the encoder [B x T_i x d_v x H x W FloatTensor]
        :param input_mask: A mask indicating the valid inputs [B x T_i ByteTensor]
        :param time_steps: The time steps corresponding to each value in the input [B x T_i LongTensor]
        :return: B x T_i x d_v x H x W FloatTensor
        """

        # Modulate the input with spatial positional encoding
        modulated_input = self.spatial_pos_enc_adder(input, time_steps)

        # Pass modulated input through the blocks
        latest_enc = modulated_input
        for n_block in xrange(self.num_blocks):
            latest_enc = self.encoder_blocks[n_block](latest_enc, input_mask)

        return latest_enc


class ConvTransformerDecoder(nn.Module):

    def __init__(self, num_blocks, num_heads, d_v, d_ff):
        """Constructor

        :param num_blocks: Number of blocks in the encoder stack
        :param num_heads: Number of heads to use for each multi-head attention module
        :param d_v: Dimensionality (number of features) of the values
        :param d_ff: Intermediate dimensionality of the "point-wise" feed-forward layers
        """

        super(ConvTransformerDecoder, self).__init__()

        self.num_blocks = num_blocks

        self.spatial_pos_enc_adder = SpatialPositionalEncodingAdder(d_v)

        self.decoder_blocks = nn.ModuleList()
        for _ in xrange(num_blocks):
            self.decoder_blocks.append(ConvTransformerDecoderBlock(num_heads, d_v=d_v, d_ff=d_v))


    def forward(self, enc_output, enc_seq_mask, init_dec_input, dec_time_steps_full, dec_prod_mask_full):
        """Forward method

        :param enc_output: The output from a Transformer encoder [B x T_i x d_v x H x W]
        :param enc_seq_mask: Mask for the encoder outputs to attend to [B x T_i]
        :param init_dec_input: The initial input to this decoder [B x T_o_old x d_v x H x W]
        :param dec_time_steps_full: The time steps corresponding to each decoder input [B x T_o]
        :param dec_prod_mask_full: Binary mask of values that can be attended to [B x T_o x T_o ByteTensor]. If entry
                                   (b, i, j) is 0, then the i'th output cannot depend on the j'th input for the b'th
                                   item.
        :return: B x T_o x d_v x H x W
        """

        T_o_old = init_dec_input.shape[1]
        T_o = dec_time_steps_full.shape[1]

        # Modulate initial decoder inputs
        modulated_init_dec_input = self.spatial_pos_enc_adder(init_dec_input, dec_time_steps_full[:, :T_o_old])

        # Pass the initial decoder inputs through the decoder
        init_dec_prod_mask = dec_prod_mask_full[:, :T_o_old, :T_o_old]
        dec_reps = [modulated_init_dec_input]
        for n_block in xrange(self.num_blocks):
            latest_dec_rep = self.decoder_blocks[n_block](dec_reps[n_block], dec_reps[n_block], enc_output,
                                                          enc_seq_mask, init_dec_prod_mask)
            dec_reps.append(latest_dec_rep)

        # Sequentially pass the latest decoder output through the decoder
        for t in xrange(T_o_old, T_o):
            # Get top-level final output
            new_dec_input = dec_reps[-1][:, -1:, :, :, :]  # B x 1 x d_v x H x W
            # Modulate new decoder input [B x 1 x d_v x H x W]
            new_modulated_dec_input = self.spatial_pos_enc_adder(new_dec_input, dec_time_steps_full[:, t:t + 1])
            # Get new product mask
            dec_prod_mask_new = dec_prod_mask_full[:, t:t+1, :t+1]  # B x 1 x t+1
            # Augment dec_reps with representations of the new decoder input
            dec_reps[0] = torch.cat([dec_reps[0], new_modulated_dec_input], dim=1)  # B x t+1 x d_v x H x W
            for n_block in xrange(self.num_blocks):
                latest_dec_rep = self.decoder_blocks[n_block](new_modulated_dec_input, dec_reps[n_block], enc_output,
                                                              enc_seq_mask, dec_prod_mask_new)
                dec_reps[n_block + 1] = torch.cat([dec_reps[n_block + 1], latest_dec_rep], dim=1)

        return dec_reps[-1]


class ConvSkipConTransformerEncoder(ConvTransformerEncoder):


    def forward(self, input, input_mask, time_steps):
        """Forward method

        :param input: The standard input to the encoder [B x T_i x d_v x H x W FloatTensor]
        :param input_mask: A mask indicating the valid inputs [B x T_i ByteTensor]
        :param time_steps: The time steps corresponding to each value in the input [B x T_i LongTensor]
        :return: num_blocks x [B x T_i x d_v x H x W FloatTensor]
        """

        # Modulate the input with spatial positional encoding
        modulated_input = self.spatial_pos_enc_adder(input, time_steps)

        # Pass modulated input through the blocks
        encs = [modulated_input]
        for n_block in xrange(self.num_blocks):
            encs.append(self.encoder_blocks[n_block](encs[-1], input_mask))

        return encs[1:]


class ConvSkipConTransformerDecoder(ConvTransformerDecoder):
    """In each level, the decoder attends to the outputs of the encoder block at the opposite level. Specifically,
    at level i in the decoder, it attends to level n-i from the encoder."""

    def forward(self, enc_outputs, enc_seq_mask, init_dec_input, dec_time_steps_full, dec_prod_mask_full):
        """Forward method

        :param enc_outputs: The outputs from a Transformer encoder (num_blocks x [B x T_i x d_v x H x W])
        :param enc_seq_mask: Mask for the encoder outputs to attend to [B x T_i]
        :param init_dec_input: The initial input to this decoder [B x T_o_old x d_v x H x W]
        :param dec_time_steps_full: The time steps corresponding to each decoder input [B x T_o]
        :param dec_prod_mask_full: Binary mask of values that can be attended to [B x T_o x T_o ByteTensor]. If entry
                                   (b, i, j) is 0, then the i'th output cannot depend on the j'th input for the b'th
                                   item.
        :return: B x T_o x d_v x H x W
        """

        assert(len(enc_outputs) == self.num_blocks)

        T_o_old = init_dec_input.shape[1]
        T_o = dec_time_steps_full.shape[1]

        # Modulate initial decoder inputs
        modulated_init_dec_input = self.spatial_pos_enc_adder(init_dec_input, dec_time_steps_full[:, :T_o_old])

        # Pass the initial decoder inputs through the decoder
        init_dec_prod_mask = dec_prod_mask_full[:, :T_o_old, :T_o_old]
        dec_reps = [modulated_init_dec_input]
        for n_block in xrange(self.num_blocks):
            latest_dec_rep = self.decoder_blocks[n_block](dec_reps[n_block], dec_reps[n_block],
                                                          enc_outputs[-n_block - 1], enc_seq_mask, init_dec_prod_mask)
            dec_reps.append(latest_dec_rep)

        # Sequentially pass the latest decoder output through the decoder
        for t in xrange(T_o_old, T_o):
            # Get top-level final output
            new_dec_input = dec_reps[-1][:, -1:, :, :, :]  # B x 1 x d_v x H x W
            # Modulate new decoder input [B x 1 x d_v x H x W]
            new_modulated_dec_input = self.spatial_pos_enc_adder(new_dec_input, dec_time_steps_full[:, t:t + 1])
            # Get new product mask
            dec_prod_mask_new = dec_prod_mask_full[:, t:t+1, :t+1]  # B x 1 x t+1
            # Augment dec_reps with representations of the new decoder input
            dec_reps[0] = torch.cat([dec_reps[0], new_modulated_dec_input], dim=1)  # B x t+1 x d_v x H x W
            for n_block in xrange(self.num_blocks):
                latest_dec_rep = self.decoder_blocks[n_block](new_modulated_dec_input, dec_reps[n_block],
                                                              enc_outputs[-n_block - 1], enc_seq_mask,
                                                              dec_prod_mask_new)
                dec_reps[n_block + 1] = torch.cat([dec_reps[n_block + 1], latest_dec_rep], dim=1)

        return dec_reps[-1]


class ConvHiddenTransformerDecoder(ConvTransformerDecoder):
    """In each level, the decoder attends to the outputs of the encoder block at the same level."""

    def forward(self, enc_outputs, enc_seq_mask, init_dec_input, dec_time_steps_full, dec_prod_mask_full):
        """Forward method

        :param enc_outputs: The outputs from a Transformer encoder (num_blocks x [B x T_i x d_v x H x W])
        :param enc_seq_mask: Mask for the encoder outputs to attend to [B x T_i]
        :param init_dec_input: The initial input to this decoder [B x T_o_old x d_v x H x W]
        :param dec_time_steps_full: The time steps corresponding to each decoder input [B x T_o]
        :param dec_prod_mask_full: Binary mask of values that can be attended to [B x T_o x T_o ByteTensor]. If entry
                                   (b, i, j) is 0, then the i'th output cannot depend on the j'th input for the b'th
                                   item.
        :return: B x T_o x d_v x H x W
        """

        assert(len(enc_outputs) == self.num_blocks)

        T_o_old = init_dec_input.shape[1]
        T_o = dec_time_steps_full.shape[1]

        # Modulate initial decoder inputs
        modulated_init_dec_input = self.spatial_pos_enc_adder(init_dec_input, dec_time_steps_full[:, :T_o_old])

        # Pass the initial decoder inputs through the decoder
        init_dec_prod_mask = dec_prod_mask_full[:, :T_o_old, :T_o_old]
        dec_reps = [modulated_init_dec_input]
        for n_block in xrange(self.num_blocks):
            latest_dec_rep = self.decoder_blocks[n_block](dec_reps[n_block], dec_reps[n_block],
                                                          enc_outputs[n_block], enc_seq_mask, init_dec_prod_mask)
            dec_reps.append(latest_dec_rep)

        # Sequentially pass the latest decoder output through the decoder
        for t in xrange(T_o_old, T_o):
            # Get top-level final output
            new_dec_input = dec_reps[-1][:, -1:, :, :, :]  # B x 1 x d_v x H x W
            # Modulate new decoder input [B x 1 x d_v x H x W]
            new_modulated_dec_input = self.spatial_pos_enc_adder(new_dec_input, dec_time_steps_full[:, t:t + 1])
            # Get new product mask
            dec_prod_mask_new = dec_prod_mask_full[:, t:t+1, :t+1]  # B x 1 x t+1
            # Augment dec_reps with representations of the new decoder input
            dec_reps[0] = torch.cat([dec_reps[0], new_modulated_dec_input], dim=1)  # B x t+1 x d_v x H x W
            for n_block in xrange(self.num_blocks):
                latest_dec_rep = self.decoder_blocks[n_block](new_modulated_dec_input, dec_reps[n_block],
                                                              enc_outputs[-n_block - 1], enc_seq_mask,
                                                              dec_prod_mask_new)
                dec_reps[n_block + 1] = torch.cat([dec_reps[n_block + 1], latest_dec_rep], dim=1)

        return dec_reps[-1]