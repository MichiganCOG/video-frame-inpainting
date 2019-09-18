import argparse
import multiprocessing
import os
import re

import moviepy.editor as mpy
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.util.util import get_folder_paths_at_depth


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


def draw_border(image, color):
    """Draws a border around the given image.

    :param image: A np.ndarray image
    :param color: The color of the border. Supports any PIL ImageColor (e.g. strings or tuples)
    """
    W, H = image.size
    image_rgb = image.convert('RGB')
    draw = ImageDraw.Draw(image_rgb)
    draw.line([(1, 0), (W-2, 0), (W-2, H-1), (1, H-1), (1, 0)], width=2, fill=color)
    return image_rgb


def get_files_in_path(root_path, file_name_pattern):
    """Return the full paths to the files under the given path whose file name matches the file name pattern.

    :param root_path: The parent directory of the files to obtain
    :param file_name_pattern: A regex pattern that matches the desired file names
    """
    filter_fn = lambda x: re.match(file_name_pattern, x) is not None
    root_path_base_names = filter(filter_fn, os.listdir(root_path))
    return sorted([os.path.join(root_path, base_name) for base_name in root_path_base_names])


def animate_frames_in_path((qual_frame_root_path, fps, high_quality, create_gt_gif)):
    gt_frames = []
    pred_frames = []

    # Process GT preceding frames
    preceding_frame_paths = get_files_in_path(qual_frame_root_path, 'gt_preceding_[0-9]+.png')
    for preceding_frame_path in preceding_frame_paths:
        gt_preceding_frame = Image.open(preceding_frame_path)
        gt_preceding_frame = draw_border(gt_preceding_frame, 'lime')
        gt_frames.append(gt_preceding_frame)
        pred_frames.append(gt_preceding_frame)

    # Process GT middle frames
    gt_middle_frame_paths = get_files_in_path(qual_frame_root_path, 'gt_middle_[0-9]+.png')
    # If GT middle frames were found, toggle flag to generate GT sequence
    if create_gt_gif:
        if len(gt_middle_frame_paths) > 0:
            for middle_frame_path in gt_middle_frame_paths:
                gt_middle_frame = Image.open(middle_frame_path)
                gt_middle_frame = draw_border(gt_middle_frame, 'red')
                gt_frames.append(gt_middle_frame)
        else:
            raise RuntimeError('Create GT GIF flag is on, but failed to find GT middle frames in %s'
                               % qual_frame_root_path)

    # Process predicted middle frames
    pred_middle_frame_paths = get_files_in_path(qual_frame_root_path, 'pred_middle_[0-9]+.png')
    for middle_frame_path in pred_middle_frame_paths:
        pred_middle_frame = Image.open(middle_frame_path)
        pred_middle_frame = draw_border(pred_middle_frame, 'red')
        pred_frames.append(pred_middle_frame)

    # Process GT following frames
    following_frame_paths = get_files_in_path(qual_frame_root_path, 'gt_following_[0-9]+.png')
    for following_frame_path in following_frame_paths:
        gt_following_frame = Image.open(following_frame_path)
        gt_following_frame = draw_border(gt_following_frame, 'lime')
        gt_frames.append(gt_following_frame)
        pred_frames.append(gt_following_frame)

    # Create GT GIF
    if create_gt_gif:
        gt_gif_path = os.path.join(qual_frame_root_path, 'gt.gif')
        create_video(gt_frames, gt_gif_path, fps, high_quality=high_quality)
    # Create predicted GIF
    pred_gif_path = os.path.join(qual_frame_root_path, 'pred_final.gif')
    create_video(pred_frames, pred_gif_path, fps, high_quality=high_quality)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('qual_results_root', type=str)
    parser.add_argument('--depth', type=int, default=1,
                        help='Depth of the folders for each video (e.g. 2 for <qual_results_root>/<action>/<video>)')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(),
                        help='Number of workers to use to generate GIFs')
    parser.add_argument('--high_quality', action='store_true',
                        help='Flag to generate higher-quality GIFs at the cost of speed')
    parser.add_argument('--fps', type=int, default=3, help='Frames per second for each GIF')
    parser.add_argument('--create_gt_gif', action='store_true', help='Flag to animate the GT frames')
    args = parser.parse_args()

    # Get the paths to the qualitative frames of each test video
    qual_frame_root_paths = get_folder_paths_at_depth(args.qual_results_root, args.depth)

    if len(qual_frame_root_paths) == 0:
        print('Failed to find any qualitative results (make sure you ran predict.py before this script). Quitting...')
        return

    print('Now animating qualitative results...')

    job_args = zip(
        qual_frame_root_paths,
        [args.fps for _ in qual_frame_root_paths],
        [args.high_quality for _ in qual_frame_root_paths],
        [args.create_gt_gif for _ in qual_frame_root_paths]
    )

    pool = multiprocessing.Pool(args.num_workers)
    job_result_iter = pool.imap_unordered(animate_frames_in_path, job_args)
    for _ in tqdm(job_result_iter, total=len(qual_frame_root_paths)):
        pass

    print('Done animating qualitative results.')


if __name__ == '__main__':
    main()
