import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.base_dataset import ContiguousVideoClipDataset, DisjointVideoClipDataset
from src.environments.environments import create_eval_environment
from src.models.create_model import create_model
from src.options.options import TestOptions
from src.util.util import makedir, listopt, to_numpy, inverse_transform, fore_transform, as_variable


def main():

    # parse test arguments and create directories
    opt = TestOptions().parse(allow_unknown=True)
    listopt(opt)

    # create dataloader
    if opt.disjoint_clips:
        test_dataset = DisjointVideoClipDataset(opt.c_dim, opt.test_video_list_path, opt.K, opt.F, opt.image_size,
                                                opt.padding_size)
    else:
        test_dataset = ContiguousVideoClipDataset(opt.c_dim, opt.test_video_list_path, opt.K + opt.T + opt.F, False,
                                                  False, opt.image_size, False, opt.padding_size)
    test_data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                  num_workers=opt.num_threads, drop_last=False)
    print('# testing videos = %d' % len(test_dataset))

    # create model
    fill_in_model = create_model(opt.model_key)
    env = create_eval_environment(fill_in_model, opt.checkpoints_dir, opt.name, opt.snapshot_file_name,
                                  opt.padding_size)

    # initialize progress bar
    pbar = tqdm(total=len(test_data_loader))

    # evaluate each video
    for i, data in enumerate(test_data_loader):
        # prepare the ground truth in the form of [batch, t, h, w, c]
        all_frames = data['targets']
        clip_label = data['clip_label']

        # compute the inpainting results
        preceding_frames = all_frames[:, :opt.K, :, :, :]
        following_frames = all_frames[:, -opt.F:, :, :, :]

        env.set_test_inputs(preceding_frames, following_frames)

        env.T = opt.T
        env.eval()
        env.forward_test()

        # save frames to disk
        pred = env.gen_output['pred']  # B x T x C x H x W
        pred_forward = env.gen_output.get('pred_forward')
        pred_backward = env.gen_output.get('pred_backward')
        interp_net_outputs_1 = env.gen_output.get('interp_net_outputs_1')
        interp_net_outputs_2 = env.gen_output.get('interp_net_outputs_2')

        for b in xrange(pred.shape[0]):
            cur_image_root_path = os.path.join(opt.qual_result_root, clip_label[b])

            # Write ground truth frames
            save_video_frames(preceding_frames[b, :, :, :opt.image_size[0], :opt.image_size[1]], cur_image_root_path,
                              'gt_preceding')
            save_video_frames(following_frames[b, :, :, :opt.image_size[0], :opt.image_size[1]], cur_image_root_path,
                              'gt_following', counter_start=opt.K+opt.T)
            if not opt.disjoint_clips:
                gt_middle_frames = all_frames[:, opt.K:-opt.F, :, :, :]
                save_video_frames(gt_middle_frames[b, :, :, :opt.image_size[0], :opt.image_size[1]],
                                  cur_image_root_path, 'gt_middle', counter_start=opt.K)

            # Write predicted middle frames
            save_video_frames(pred[b, :, :, :opt.image_size[0], :opt.image_size[1]].data, cur_image_root_path,
                              'pred_middle', counter_start=opt.K)

            # Write intermediate predictions
            if opt.intermediate_preds:
                if pred_forward is not None:
                    save_video_frames(pred_forward[b, :, :, :opt.image_size[0], :opt.image_size[1]].data,
                                      cur_image_root_path, 'pred_middle_forward', counter_start=opt.K)
                if pred_backward is not None:
                    save_video_frames(pred_backward[b, :, :, :opt.image_size[0], :opt.image_size[1]].data,
                                      cur_image_root_path, 'pred_middle_backward', counter_start=opt.K)
                if interp_net_outputs_1 is not None:
                    save_video_frames(interp_net_outputs_1[b, :, :, :opt.image_size[0], :opt.image_size[1]].data,
                                      cur_image_root_path, 'interp_net_outputs_1', counter_start=opt.K)
                if interp_net_outputs_2 is not None:
                    save_video_frames(interp_net_outputs_2[b, :, :, :opt.image_size[0], :opt.image_size[1]].data,
                                      cur_image_root_path, 'interp_net_outputs_2', counter_start=opt.K)

        pbar.update()

    pbar.close()
    print('Done.')


def save_video_frames(video, image_root_dir, image_name_prefix, counter_start=0):
    """Saves the frames in the given video to a folder.

    :param video: T x C x H x W FloatTensor, range in [-1, 1]. If C == 3, input should be in BGR color space
    :param image_root_dir: The directory where the video frames should be saved
    ;param image_name_prefix: The string used to prefix each video frame
    :param counter_start: The starting index used to name each video frame
    """

    T, C, H, W = video.shape
    clipped_video = torch.clamp(video, -1, 1)
    makedir(image_root_dir)
    for t in xrange(T):
        image_path = os.path.join(image_root_dir, '%s_%04d.png' % (image_name_prefix, t + counter_start))
        frame_np = to_numpy(clipped_video[t, :, :, :], transpose=(1, 2, 0))
        frame_np_uint8 = (255 * inverse_transform(frame_np)).astype(np.uint8)
        pil_image = Image.fromarray(frame_np_uint8[:, :, 0] if C == 1 else frame_np_uint8[:, :, ::-1])
        pil_image.save(image_path)

if __name__ == '__main__':
    main()
