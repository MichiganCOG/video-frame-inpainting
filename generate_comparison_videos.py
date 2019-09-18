import argparse
import os
import re

import moviepy.editor as mpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.util.util import makedir


__FONT__ = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 30)
__BORDER_SIZE__ = 5


def create_video(frames, save_path, fps, high_quality=False):
    """Save the given frames as a GIF.

    :param frames: A sequence of PIL Images
    :param save_path: The path to the final GIF file
    :param fps: The frame rate of the GIF
    :param high_quality: Flag to optimize GIF quality with ImageMagick backend (at the cost of speed)
    """

    clip = mpy.ImageSequenceClip([np.array(frame) for frame in frames], fps=fps)
    if high_quality:
        clip.write_gif(save_path, verbose=False, progress_bar=False, program='ImageMagick', opt='optimizeplus')
    else:
        clip.write_gif(save_path, verbose=False, progress_bar=False)


def main(results_root, exp_names, exp_labels, clip_names, save_root):
    makedir(save_root)
    if exp_labels is not None:
        assert len(exp_names) == len(exp_labels)
    else:
        exp_labels = exp_names
    frame_root_paths = [os.path.join(results_root, 'images', x) for x in exp_names]

    for clip_name in clip_names:
        clip_paths = [os.path.join(x, clip_name) for x in frame_root_paths]

        # Get all the frame file names
        first_clip_path = clip_paths[0]
        frame_names = filter(lambda x: re.match('((gt)|(pred))_((preceding)|(middle)|(following))_[0-9]+\.png', x),
                             os.listdir(first_clip_path))
        preceding_frame_names = sorted(filter(lambda x: x.startswith('gt_preceding'), frame_names))
        gt_middle_frame_names = sorted(filter(lambda x: x.startswith('gt_middle'), frame_names))
        following_frame_names = sorted(filter(lambda x: x.startswith('gt_following'), frame_names))
        pred_middle_frame_names = sorted(filter(lambda x: x.startswith('pred_middle'), frame_names))

        output_frames = []
        for frame_name in preceding_frame_names:
            output_frame = generate_frame(clip_paths, exp_labels, exp_names, first_clip_path, frame_name, 'green',
                                          frame_name)
            output_frames.append(output_frame)
        for pred_frame_name, gt_frame_name in zip(pred_middle_frame_names, gt_middle_frame_names):
            output_frame = generate_frame(clip_paths, exp_labels, exp_names, first_clip_path, pred_frame_name, 'yellow',
                                          gt_frame_name)
            output_frames.append(output_frame)
        for frame_name in following_frame_names:
            output_frame = generate_frame(clip_paths, exp_labels, exp_names, first_clip_path, frame_name, 'green',
                                          frame_name)
            output_frames.append(output_frame)

        create_video(output_frames, os.path.join(save_root, '{}.gif'.format(clip_name)), 3, high_quality=True)


def generate_frame(clip_paths, exp_labels, exp_names, first_clip_path, frame_name, border_color, gt_frame_name):
    # Get the frame from each experiment result
    exp_frame_paths = [os.path.join(x, frame_name) for x in clip_paths]
    frames = [Image.open(x).convert('RGB') for x in exp_frame_paths]
    frames_bordered = [ImageOps.expand(x, border=__BORDER_SIZE__, fill=border_color) for x in frames]
    im_w, im_h = frames_bordered[0].size
    output_frame = Image.new('RGB', ((len(exp_names) + 1) * im_w, im_h + 30))
    draw = ImageDraw.Draw(output_frame)
    for i, frame in enumerate(frames_bordered):
        output_frame.paste(frame, (i * im_w, 0))
        exp_label_size = __FONT__.getsize(exp_labels[i])
        draw.text((i * im_w + (im_w - exp_label_size[0]) / 2, im_h), exp_labels[i], font=__FONT__)
    gt_bordered = ImageOps.expand(Image.open(os.path.join(first_clip_path, gt_frame_name)).convert('RGB'),
                                  border=__BORDER_SIZE__, fill=border_color)
    output_frame.paste(gt_bordered, ((i + 1) * im_w, 0))
    exp_label_size = __FONT__.getsize('GT')
    draw.text(((i + 1) * im_w + (im_w - exp_label_size[0]) / 2, im_h), 'GT', font=__FONT__)
    return output_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', type=str, required=True,
                        help='Path to the results for the dataset (should contain "images" and "quantitative"')
    parser.add_argument('--exp_names', type=str, nargs='+', required=True,
                        help='Experiment names for the desired models')
    parser.add_argument('--exp_labels', type=str, nargs='+', default=None,
                        help='Alternative labels for the desired models. Length must match exp_names')
    parser.add_argument('--clip_names', type=str, nargs='+', required=True,
                        help='Names of the clips to compare')
    parser.add_argument('--save_root', type=str, required=True,
                        help='Path to save GIFs')
    args = parser.parse_args()

    main(**vars(args))