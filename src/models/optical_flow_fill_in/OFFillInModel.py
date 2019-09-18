import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .optical_flow_utils import interpolate_frames
from ...util.util import fore_transform, inverse_transform

class OFFillInModel(nn.Module):

    def forward(self, T, preceding_frames, following_frames):
        B, _, C, H, W = preceding_frames.shape

        last_preceding_frame = preceding_frames[:, -1]
        first_following_frame = following_frames[:, 0]

        # Convert to np.ndarrays
        last_preceding_frame_np = last_preceding_frame.data.cpu().numpy()
        first_following_frame_np = first_following_frame.data.cpu().numpy()
        # Convert to uint8
        last_preceding_frame_np = (255 * inverse_transform(last_preceding_frame_np)).astype(np.uint8)
        first_following_frame_np = (255 * inverse_transform(first_following_frame_np)).astype(np.uint8)

        all_fill_in_frames = []

        for b in xrange(B):
            # Move channel dimension to end
            last_preceding_frame_cur_b = last_preceding_frame_np[b].transpose(1, 2, 0)
            first_following_frame_cur_b = first_following_frame_np[b].transpose(1, 2, 0)
            # Convert to BGR
            if C == 1:
                last_preceding_frame_cur_b = cv2.cvtColor(last_preceding_frame_cur_b, cv2.COLOR_GRAY2BGR)
                first_following_frame_cur_b = cv2.cvtColor(first_following_frame_cur_b, cv2.COLOR_GRAY2BGR)
            else:
                last_preceding_frame_cur_b = cv2.cvtColor(last_preceding_frame_cur_b, cv2.COLOR_RGB2BGR)
                first_following_frame_cur_b = cv2.cvtColor(first_following_frame_cur_b, cv2.COLOR_RGB2BGR)

            # T x [H x W x 3]
            fill_in_frames_cur_b = interpolate_frames(last_preceding_frame_cur_b, first_following_frame_cur_b, T)
            # Convert each frame to RGB or grayscale
            if C == 1:
                fill_in_frames_cur_b = [cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis] \
                                        for a in fill_in_frames_cur_b]
            else:
                fill_in_frames_cur_b = [cv2.cvtColor(a, cv2.COLOR_BGR2RGB) for a in fill_in_frames_cur_b]

            all_fill_in_frames.append(np.stack(fill_in_frames_cur_b))

        # Combine all frames into one np.ndarray
        all_fill_in_frames = np.stack(all_fill_in_frames)  # B x T x H x W x C
        # Remap from [0, 255] to [-1, 1]
        all_fill_in_frames = fore_transform(all_fill_in_frames / 255.)
        # Move channel dimension ahead of H x W
        all_fill_in_frames = all_fill_in_frames.transpose((0, 1, 4, 2, 3))
        fill_in_frames_var = Variable(torch.from_numpy(all_fill_in_frames), requires_grad=False)

        return {
            'pred': fill_in_frames_var
        }