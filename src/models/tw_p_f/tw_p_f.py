import numpy as np
import torch
import torch.nn as nn


class TimeWeightedPFFillInModel(nn.Module):
    """Computes the middle frames as weighted averages between the last preceding and the first following frame.

    For middle frame t (1-indexed) within T middle frames, the weight of the first following frame is t/(T+1); the
    weight of the last preceding frame is 1 minus that.
    """

    def forward(self, T, preceding_frames, following_frames):
        """Forward method

        :param T: The number of middle frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        """

        # Get the last preceding and first following frame
        last_p_frames = preceding_frames[:, -1:, :, :, :]
        first_f_frames = following_frames[:, :1, :, :, :]
        # Compute weight of following frame per time step
        w = np.linspace(0, 1, num=T+2).tolist()[1:-1]

        pred_middle_frames = []
        for t in range(T):
            pred_middle_frames_cur_t = (1 - w[t]) * last_p_frames + w[t] * first_f_frames
            pred_middle_frames.append(pred_middle_frames_cur_t)

        return {
            'pred': torch.cat(pred_middle_frames, dim=1)
        }