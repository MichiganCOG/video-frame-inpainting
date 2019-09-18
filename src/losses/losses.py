import torch.nn as nn


class GDL(nn.Module):
    """Image gradient difference loss as defined by Mathieu et al. (https://arxiv.org/abs/1511.05440)."""

    def __init__(self, reduce=True):
        """Constructor

        :param reduce: Whether to mean-reduce the loss term across all dimensions
        """
        super(GDL, self).__init__()

        self.reduce = reduce
        self._l1_loss = nn.L1Loss(reduce=False)

    def forward(self, input, target):
        """Forward method

        :param input: The predicted value [B x ... x H x W]
        :param target: The actual value [B x ... x H x W]
        :return: B x ... x H-1 x W-1 FloatTensor if reduce is True, else single-element FloatTensor
        """
        B = input.size(0)
        H, W = input.shape[-2:]
        other_dims = input.shape[:-2]
        input_flat = input.view(-1, H, W)
        target_flat = target.view(-1, H, W)

        input_w_grad = input_flat[:, :, :-1] - input_flat[:, :, 1:]
        input_h_grad = input_flat[:, 1:, :] - input_flat[:, :-1, :]
        target_w_grad = target_flat[:, :, :-1] - target_flat[:, :, 1:]
        target_h_grad = target_flat[:, 1:, :] - target_flat[:, :-1, :]
        w_grad_loss_flat = self._l1_loss(input_w_grad, target_w_grad)[:, 1:, :]
        h_grad_loss_flat = self._l1_loss(input_h_grad, target_h_grad)[:, :, 1:]

        new_shape = other_dims + (H - 1, W - 1)
        w_grad_loss = w_grad_loss_flat.contiguous().view(*new_shape)
        h_grad_loss = h_grad_loss_flat.contiguous().view(*new_shape)

        loss = w_grad_loss + h_grad_loss
        if self.reduce:
            return loss.view(B, -1).mean()
        else:
            return loss