from __future__ import division

import argparse
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from glob import glob
from warnings import warn

from PIL import Image, ImageDraw, ImageFont

from src.util.vis_utils import in2cm, add_image_to_pdf, add_cropped_image_to_pdf, add_text_to_pdf, get_text_width, \
    create_pdf, GREEN, YELLOW, WHITE, ORANGE

__SCRIPT_DIR__ = os.path.abspath(os.path.dirname(__file__))
__LABEL_FONT__ = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 40)
__SMALL_FONT__ = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 30)


def create_video(clip_names, dataset_name, dest_path, exp_names, model_labels, results_root, total_num_frames,
                 video_type, pool):
    assert(video_type in ['gif', 'mp4', 'mp4-uncomp', 'images', 'images-uncomp'])

    frame_width = 320
    frame_height = 240
    border_size = 4
    padding_size = 4
    label_frame_padding_size = 20
    fps = 3

    # Get maximum width of all labels
    max_label_width = -1
    for label in model_labels + ['Ground truth']:
        max_label_width = max(max_label_width, __LABEL_FONT__.getsize(label)[0])

    canvas_width = max_label_width + label_frame_padding_size + len(clip_names) * frame_width + 2 * len(clip_names) * border_size \
                   + (len(clip_names) - 1) * padding_size
    canvas_height = (len(model_labels) + 1) * frame_height + 2 * (len(model_labels) + 1) * border_size \
                    + len(model_labels) * padding_size

    temp_dir = tempfile.mkdtemp()

    # Save video frames to temporary directory in parallel
    save_video_frame_args = [(border_size, canvas_height, canvas_width, clip_names, dataset_name, exp_names, frame_height,
                             frame_width, label_frame_padding_size, max_label_width, model_labels, padding_size,
                             results_root, t, temp_dir) for t in xrange(total_num_frames)]
    pool.map(save_video_frame, save_video_frame_args)

    if video_type == 'images':
        for t in xrange(total_num_frames):
            cur_frame = Image.open(os.path.join(temp_dir, '%02d.png' % t))
            cur_frame.save(os.path.join(dest_path, '%02d.jpg' % t))
    if video_type == 'images-uncomp':
        shutil.copytree(temp_dir, os.path.join(dest_path, 'video_frames'))
    if video_type == 'mp4':
        ffmpeg_cmd = ['ffmpeg', '-r', str(fps), '-i', '{temp_dir}/%02d.png'.format(temp_dir=temp_dir),
                      '-c:v', 'libx264', '{dest_path}/video.mp4'.format(dest_path=dest_path), '-y']
        subprocess.call(ffmpeg_cmd)
    elif video_type == 'mp4-uncomp':
        ffmpeg_cmd = ['ffmpeg', '-r', str(fps), '-i', '{temp_dir}/%02d.png'.format(temp_dir=temp_dir),
                      '-c:v', 'libx264', '-crf', str(0), '{dest_path}/video.mp4'.format(dest_path=dest_path), '-y']
        subprocess.call(ffmpeg_cmd)
    elif video_type == 'gif':
        ffmpeg_cmd = ['ffmpeg', '-r', str(fps), '-i', '{temp_dir}/%02d.png'.format(temp_dir=temp_dir),
                      '{dest_path}/video.gif'.format(dest_path=dest_path), '-y']
        subprocess.call(ffmpeg_cmd)

    shutil.rmtree(temp_dir)


def save_video_frame((border_size, canvas_height, canvas_width, clip_names, dataset_name, exp_names, frame_height,
                     frame_width, label_frame_padding_size, max_label_width, model_labels, padding_size,
                     results_root, t, temp_dir)):

    # Create canvas and drawing context
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=WHITE)
    draw = ImageDraw.Draw(canvas)
    # Draw current frame number
    draw.text((0, 0), 'Frame %02d' % t, fill=0, font=(__SMALL_FONT__))
    # Draw labels
    for l, label in enumerate(model_labels):
        draw.text((0, l * (2 * border_size + padding_size + frame_height) + frame_height // 2), label,
                  font=(__LABEL_FONT__), fill=0)
    draw.text((0, len(model_labels) * (2 * border_size + padding_size + frame_height) + frame_height // 2),
              'Ground truth', font=(__LABEL_FONT__), fill=0)
    for e, exp_name in enumerate(exp_names):
        for c, clip_name in enumerate(clip_names):
            # Draw video frame
            images_root = os.path.join(results_root, dataset_name, 'images', exp_name, clip_name)
            if not os.path.isdir(images_root):
                warn('Could not find image folder %s. Images will not be printed to video' % images_root)
            # Find the frame for the current time step
            possible_frame_names = ['gt_preceding_%04d.png' % t,
                                    'pred_middle_%04d.png' % t,
                                    'gt_following_%04d.png' % t]
            frame = None
            for frame_name in possible_frame_names:
                image_path = os.path.join(images_root, frame_name)
                if os.path.isfile(image_path):
                    frame = Image.open(image_path)
                    break
            if frame is None:
                warn('Could not find valid frame for time step %d in %s. Image will not be printed to video'
                     % (t, images_root))
                continue
            canvas.paste(frame, (max_label_width + label_frame_padding_size + border_size
                                 + c * (2 * border_size + padding_size + frame_width),
                                 border_size + e * (2 * border_size + padding_size + frame_height)))

            # Draw border
            for b in xrange(border_size):
                border_color = YELLOW if 'middle' in frame_name else GREEN
                draw.rectangle([(max_label_width + label_frame_padding_size + border_size
                                 + c * (2 * border_size + padding_size + frame_width) - b - 1,
                                 border_size - 1 + e * (2 * border_size + padding_size + frame_height) - b),
                                (max_label_width + label_frame_padding_size + border_size
                                 + c * (2 * border_size + padding_size + frame_width) + frame_width + b,
                                 border_size + frame_height + e * (2 * border_size + padding_size + frame_height) + b)],
                               outline=border_color)
    e = len(exp_names)
    for c, clip_name in enumerate(clip_names):
        # Draw video frame
        images_root = os.path.join(results_root, dataset_name, 'images', exp_name, clip_name)
        if not os.path.isdir(images_root):
            warn('Could not find image folder %s. Images will not be printed to video' % images_root)
        # Find the frame for the current time step
        possible_frame_names = ['gt_preceding_%04d.png' % t, 'gt_middle_%04d.png' % t, 'gt_following_%04d.png' % t]
        frame = None
        for frame_name in possible_frame_names:
            image_path = os.path.join(images_root, frame_name)
            if os.path.isfile(image_path):
                frame = Image.open(image_path)
                break
        if frame is None:
            warn('Could not find valid frame for time step %d in %s. Image will not be printed to video'
                 % (t, images_root))
            continue
        canvas.paste(frame, (max_label_width + label_frame_padding_size + border_size
                             + c * (2 * border_size + padding_size + frame_width),
                             border_size + e * (2 * border_size + padding_size + frame_height)))

        # Draw border
        for b in xrange(border_size):
            border_color = YELLOW if 'middle' in frame_name else GREEN
            draw.rectangle([(max_label_width + label_frame_padding_size + border_size
                             + c * (2 * border_size + padding_size + frame_width) - b - 1,
                             border_size - 1 + e * (2 * border_size + padding_size + frame_height) - b), (
                             max_label_width + label_frame_padding_size + border_size
                             + c * (2 * border_size + padding_size + frame_width) + frame_width + b,
                             border_size + frame_height + e * (2 * border_size + padding_size + frame_height) + b)],
                           outline=border_color)

    # Save frame to temporary folder
    canvas.save(os.path.join(temp_dir, '%02d.png' % t))


def create_pdfs(clip_names, dataset_name, dest_path, exp_names, model_labels, results_root, frame_indexes,
                frame_width_cm, frame_height_cm, border_width_cm, spacing_cm, font_size_pt, label_frame_width_cm,
                zoom_regions, zoom_window_positions):

    # Determine longest width of rendered labels
    max_text_width_cm = -1
    for model_label in (model_labels + ['Ground truth']):
        max_text_width_cm = max(max_text_width_cm, get_text_width(model_label, font_size_pt))

    # Compute height and width of final figures
    fig_width_cm = max_text_width_cm + label_frame_width_cm + len(frame_indexes) * (
        2 * border_width_cm + frame_width_cm + spacing_cm) - spacing_cm
    fig_height_cm = (len(exp_names) + 1) * (2 * border_width_cm + frame_height_cm + spacing_cm) - spacing_cm
    # Compute width of model label plus some padding
    for zoom_region, zoom_window_position, clip_name in zip(zoom_regions, zoom_window_positions, clip_names):
        # Create PDF page with specified size
        print('Creating PDF of size %f x %f cm' % (fig_width_cm, fig_height_cm))
        pdf = create_pdf(fig_width_cm, fig_height_cm, 'cm')

        for i, (exp_name, model_label) in enumerate(zip(exp_names, model_labels)):
            y_cm = i * (frame_height_cm + 2 * border_width_cm + spacing_cm)
            y_offset = (frame_height_cm + 2 * border_width_cm - 0.7 * in2cm(font_size_pt / 72)) / 2
            cur_text_width_cm = get_text_width(model_label, font_size_pt)
            add_text_to_pdf(pdf, model_label, max_text_width_cm - cur_text_width_cm, y_cm + y_offset, font_size_pt)

            images_root = os.path.join(results_root, dataset_name, 'images', exp_name, clip_name)
            if not os.path.isdir(images_root):
                warn('Could not find image folder %s. Images will not be printed to PDF' % images_root)

            # Compute number of preceding, middle, and following frames from the file names in images_root
            K = len(glob(os.path.join(images_root, 'gt_preceding_*')))
            T = len(glob(os.path.join(images_root, 'gt_middle_*')))
            F = len(glob(os.path.join(images_root, 'gt_following_*')))
            if max(frame_indexes) >= K + T + F:
                continue
            # assert (max(frame_indexes) < K + T + F)

            preceding_frame_names = ['gt_preceding_%04d.png' % t for t in xrange(K)]
            gt_middle_frame_names = ['gt_middle_%04d.png' % t for t in xrange(K, K + T)]
            following_frame_names = ['gt_following_%04d.png' % t for t in xrange(K + T, K + T + F)]
            pred_middle_frame_names = ['pred_middle_%04d.png' % t for t in xrange(K, K + T)]

            image_names = preceding_frame_names + pred_middle_frame_names + following_frame_names
            image_paths = [os.path.join(images_root, image_name) for image_name in image_names]

            for j, frame_index in enumerate(frame_indexes):
                x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + j * (
                frame_width_cm + 2 * border_width_cm + spacing_cm)
                y_cm = border_width_cm + i * (frame_height_cm + 2 * border_width_cm + spacing_cm)
                add_image_to_pdf(pdf, image_paths[frame_index], x_cm, y_cm, frame_width_cm, frame_height_cm, b_cm=border_width_cm,
                                 color=(YELLOW if K <= frame_index < K + T else GREEN))

                if zoom_region is not None and zoom_window_position is not None:
                    # Redraw the zoomed region with a border around it
                    # Note: Redrawing makes the border consistent with rest of figure
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + j * (
                    frame_width_cm + 2 * border_width_cm + spacing_cm) + zoom_region[0] * frame_width_cm
                    y_cm = border_width_cm + i * (frame_height_cm + 2 * border_width_cm + spacing_cm) \
                           + zoom_region[1] * frame_height_cm
                    add_cropped_image_to_pdf(pdf, image_paths[frame_index], zoom_region, x_cm, y_cm,
                                             (zoom_region[2] - zoom_region[0]) * frame_width_cm,
                                             (zoom_region[3] - zoom_region[1]) * frame_height_cm,
                                             b_cm=border_width_cm, color=ORANGE)

                    # Draw the zoomed region in the specified position
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + j * (
                    frame_width_cm + 2 * border_width_cm + spacing_cm) + zoom_window_position[0] * frame_width_cm
                    y_cm = border_width_cm + i * (frame_height_cm + 2 * border_width_cm + spacing_cm) \
                           + zoom_window_position[1] * frame_height_cm
                    add_cropped_image_to_pdf(pdf, image_paths[frame_index], zoom_region, x_cm, y_cm,
                                             (zoom_window_position[2] - zoom_window_position[0]) * frame_width_cm,
                                             (zoom_window_position[3] - zoom_window_position[1]) * frame_height_cm,
                                             b_cm=border_width_cm, color=ORANGE)

        i = len(exp_names)
        model_label = 'Ground truth'
        y_cm = i * (frame_height_cm + 2 * border_width_cm + spacing_cm)
        y_offset = (frame_height_cm + 2 * border_width_cm - 0.7 * in2cm(font_size_pt / 72)) / 2
        cur_text_width_cm = get_text_width(model_label, font_size_pt)
        add_text_to_pdf(pdf, model_label, max_text_width_cm - cur_text_width_cm, y_cm + y_offset, font_size_pt)

        image_names = preceding_frame_names + gt_middle_frame_names + following_frame_names
        image_paths = [os.path.join(images_root, image_name) for image_name in image_names]

        for j, frame_index in enumerate(frame_indexes):
            x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + j * (
            frame_width_cm + 2 * border_width_cm + spacing_cm)
            y_cm = border_width_cm + i * (frame_height_cm + 2 * border_width_cm + spacing_cm)
            add_image_to_pdf(pdf, image_paths[frame_index], x_cm, y_cm, frame_width_cm, frame_height_cm, b_cm=border_width_cm,
                             color=GREEN)

            if zoom_region is not None and zoom_window_position is not None:

                # Redraw the zoomed region with a border around it
                # Note: Redrawing makes the border consistent with rest of figure
                x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + j * (
                frame_width_cm + 2 * border_width_cm + spacing_cm) + zoom_region[0] * frame_width_cm
                y_cm = border_width_cm + i * (frame_height_cm + 2 * border_width_cm + spacing_cm) \
                       + zoom_region[1] * frame_height_cm
                add_cropped_image_to_pdf(pdf, image_paths[frame_index], zoom_region, x_cm, y_cm,
                                         (zoom_region[2] - zoom_region[0]) * frame_width_cm,
                                         (zoom_region[3] - zoom_region[1]) * frame_height_cm,
                                         b_cm=border_width_cm, color=ORANGE)

                # Draw cropped image with border
                x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + j * (
                frame_width_cm + 2 * border_width_cm + spacing_cm) + zoom_window_position[0] * frame_width_cm
                y_cm = border_width_cm + i * (frame_height_cm + 2 * border_width_cm + spacing_cm) \
                       + zoom_window_position[1] * frame_height_cm
                add_cropped_image_to_pdf(pdf, image_paths[frame_index], zoom_region, x_cm, y_cm,
                                         (zoom_window_position[2] - zoom_window_position[0]) * frame_width_cm,
                                         (zoom_window_position[3] - zoom_window_position[1]) * frame_height_cm,
                                         b_cm=border_width_cm, color=ORANGE)

        pdf.output(os.path.join(dest_path, '%s.pdf' % clip_name))


def create_pdfs_one_middle_frame(clip_names, dataset_name, dest_path, exp_names, model_labels, results_root,
                                 frame_indexes, frame_width_cm, frame_height, border_width, spacing,
                                 font_size_pt, label_frame_width_cm, zoom_regions, zoom_window_positions):

    # This figure will only depict three frames at the top
    assert(len(frame_indexes) == 3)
    # This figure will only depict three methods (plus one ground truth)
    assert(len(exp_names) == 3 and len(model_labels) == 3)

    # Determine longest width of rendered labels
    max_text_width_cm = -1
    for model_label in (['Ground truth'] + model_labels):
        max_text_width_cm = max(max_text_width_cm, get_text_width(model_label, font_size_pt))

    # Compute width of model label plus some padding
    for zoom_region, zoom_window_position, clip_name in zip(zoom_regions, zoom_window_positions, clip_names):
        if zoom_region is None or zoom_window_position is None:
            raise NotImplementedError('Must input zoom_region and zoom_window_position')

        cropped_frame_width = (zoom_window_position[2] - zoom_window_position[0]) * frame_width_cm
        cropped_frame_height = (zoom_window_position[3] - zoom_window_position[1]) * frame_height

        # Determine column width for ground-truth frames
        col_a_width = 2 * border_width + frame_width_cm
        # Determine column width for zoomed-in regions
        col_b_width = max(max_text_width_cm, 2 * border_width + cropped_frame_width)
        # Determine row height for ground-truth frames
        row_a_height = 2 * border_width + frame_height
        # Determine row height for zoomed-in regions
        row_b_height = 2 * border_width + cropped_frame_height + spacing + .91 * in2cm(font_size_pt / 72)

        # Compute height and width of final figures
        fig_width_cm = col_a_width + 2 * col_b_width + 2 * spacing
        fig_height_cm = 3 * row_a_height + 2 * spacing

        # Compute vertical spacing between zoomed-in region rows
        row_b_spacing = (fig_height_cm - 2 * row_b_height) / 3

        # Create PDF page with specified size
        print('Creating PDF of size %f x %f cm' % (fig_width_cm, fig_height_cm))
        pdf = create_pdf(fig_width_cm, fig_height_cm, 'cm')

        for i, (exp_name, model_label) in enumerate(zip(exp_names + ['GT'], model_labels + ['Ground truth'])):
            label_text_width = get_text_width(model_label, font_size_pt)
            x_label_text_offset = (col_b_width - label_text_width) / 2
            if i % 2 == 0:
                x_cm = col_a_width + spacing
            else:
                x_cm = col_a_width + col_b_width + 2 * spacing
            y_cm = (i // 2) * row_b_height + (i // 2 + 1) * row_b_spacing
            y_label_text_offset = 2 * border_width + cropped_frame_height + spacing
            add_text_to_pdf(pdf, model_label, x_cm + x_label_text_offset, y_cm + y_label_text_offset, font_size_pt)

            images_root = os.path.join(results_root, dataset_name, 'images',
                                       exp_name if exp_name != 'GT' else exp_names[0], clip_name)
            if not os.path.isdir(images_root):
                warn('Could not find image folder %s. Images will not be printed to PDF' % images_root)

            middle_frame_image_path = os.path.join(images_root,
                                                   'pred_middle_%04d.png' % frame_indexes[1] if exp_name != 'GT'
                                                   else 'gt_middle_%04d.png' % frame_indexes[1])
            if not os.path.isfile(middle_frame_image_path):
                warn('Failed to find frame at %s, skipping' % middle_frame_image_path)
                continue

            # Draw the zoomed region
            x_image_offset = (col_b_width - 2 * border_width - cropped_frame_width) / 2 + border_width
            y_image_offset = border_width
            add_cropped_image_to_pdf(pdf, middle_frame_image_path, zoom_region, x_cm + x_image_offset,
                                     y_cm + y_image_offset, cropped_frame_width, cropped_frame_height,
                                     b_cm=border_width, color=ORANGE)

        ### DRAW FULL GROUND-TRUTH FRAMES ###

        # Compute number of preceding, middle, and following frames from the file names in images_root
        K = len(glob(os.path.join(images_root, 'gt_preceding_*')))
        T = len(glob(os.path.join(images_root, 'gt_middle_*')))
        F = len(glob(os.path.join(images_root, 'gt_following_*')))
        if max(frame_indexes) >= K + T + F:
            continue
        # assert (max(frame_indexes) < K + T + F)

        preceding_frame_names = ['gt_preceding_%04d.png' % t for t in xrange(K)]
        gt_middle_frame_names = ['gt_middle_%04d.png' % t for t in xrange(K, K + T)]
        following_frame_names = ['gt_following_%04d.png' % t for t in xrange(K + T, K + T + F)]

        image_names = preceding_frame_names + gt_middle_frame_names + following_frame_names
        image_paths = [os.path.join(images_root, image_name) for image_name in image_names]

        for j, frame_index in enumerate(frame_indexes):
            x_cm = border_width
            y_cm = border_width + j * (frame_height + 2 * border_width + spacing)
            add_image_to_pdf(pdf, image_paths[frame_index], x_cm, y_cm, frame_width_cm, frame_height, b_cm=border_width,
                             color=GREEN if frame_index != frame_indexes[1] else YELLOW)

            # Draw border around zoomed-in region on the middle frame
            if frame_index == frame_indexes[1]:
                x_cm = border_width \
                       + zoom_region[0] * frame_width_cm
                y_cm = border_width + j * (frame_height + 2 * border_width + spacing) + zoom_region[1] * frame_height
                add_cropped_image_to_pdf(pdf, image_paths[frame_index], zoom_region, x_cm, y_cm,
                                         (zoom_region[2] - zoom_region[0]) * frame_width_cm,
                                         (zoom_region[3] - zoom_region[1]) * frame_height,
                                         b_cm=border_width, color=ORANGE)

        pdf.output(os.path.join(dest_path, '%s.pdf' % clip_name))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', type=str, default='results',
                        help='Path to the "results" folder containing all experimental results')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the folder directly under "results" (e.g. "UCF-test_data_list_T=3")')
    parser.add_argument('--total_num_frames', type=int, default=None,
                        help='Total number of frames to show')
    parser.add_argument('--clip_names', type=str, nargs='+', required=True,
                        help='Names of the video clips to visualize (e.g. "v_Archery_g05_c04.avi_1-11")')
    parser.add_argument('--exp_names', type=str, nargs='+', required=True,
                        help='Folder names of one or more experiments (e.g. TAI_0200)')
    parser.add_argument('--model_labels', type=str, nargs='+', default=None,
                        help='Labels for the given experiments (e.g. "TAI")')
    parser.add_argument('--dest_path', type=str, default=os.path.join('visual_results', str(datetime.now())),
                        help='Path to place the generated PDFs')
    parser.add_argument('--video_type', type=str, choices=['gif', 'mp4', 'mp4-uncomp', 'images', 'images-uncomp'],
                        default=None,
                        help='The format of the video output. "gif" and "mp4" create GIFs and MP4s; "mp4-uncomp" '
                             'creates an uncompressed MP4 file; and "images" creates a folder of video frames. If '
                             'this option is not specified, no video/image set will be created')
    parser.add_argument('--no_pdf', action='store_true', help='Flag to disable generation of the results as a PDF')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(),
                        help='Number of workers to use when generating video frames')
    parser.add_argument('--frame_indexes', type=int, nargs='+', default=None, help='The specific frames to show')
    parser.add_argument('--pdf_frame_width', type=float, default=1.2, help='Width of each video frame in cm')
    parser.add_argument('--pdf_frame_height', type=float, default=0.9, help='Height of each video frame in cm')
    parser.add_argument('--pdf_border_width', type=float, default=0.04,
                        help='Thickness of the border around each video frame in cm')
    parser.add_argument('--pdf_spacing', type=float, default=0.05,
                        help='Thickness of margin between each video frame in cm')
    parser.add_argument('--pdf_font_size', type=int, default=8, help='Font size for the PDF text')
    parser.add_argument('--pdf_label_frame_width', type=float, default=0.5,
                        help='Size of the margin between the labels and the video frames in cm')
    parser.add_argument('--pdf_zoom_region', type=float, nargs=4, action='append', default=None,
                        help='The patch to zoom in on defined as four decimal values specifying TL-x, TL-y, BR-x, '
                             'and BR-y coordinates (TL = top left, BR = bottom right). Use this flag multiple times '
                             'to specify a different region for each clip')
    parser.add_argument('--pdf_zoom_window_position', type=float, nargs=4, action='append', default=None,
                        help='The location of the zoomed patch defined as four decimal values specifying TL-x, TL-y, '
                             'BR-x, and BR-y coordinates (TL = top left, BR = bottom right). Use this flag multiple '
                             'times to specify a different position for each clip')
    parser.add_argument('--pdf_one_middle_frame', action='store_true',
                        help='Generate a special PDF that only compares one middle frame')

    args = parser.parse_args()

    if args.frame_indexes and args.total_num_frames:
        warn('Setting both --frame_indexes and --total_num_frames may result in unexpected behavior')
    if not args.frame_indexes and not args.total_num_frames:
        raise ValueError('Must specify either --frame_indexes or --total_num_frames')
    if args.frame_indexes is not None:
        args.total_num_frames = len(args.frame_indexes)
    elif args.total_num_frames is not None:
        args.frame_indexes = range(args.total_num_frames)

    if (args.pdf_zoom_region is None) ^ (args.pdf_zoom_window_position is None):
        raise ValueError('pdf_zoom_region and pdf_zoom_window_position must be specified together or not at all')
    # Set zoom info to lists of None if not specified
    if args.pdf_zoom_region is None and args.pdf_zoom_window_position is None:
        args.pdf_zoom_region = [None for _ in args.clip_names]
        args.pdf_zoom_window_position = [None for _ in args.clip_names]
    if args.pdf_zoom_region is not None and len(args.pdf_zoom_region) != len(args.clip_names):
        raise ValueError('Number of regions specified by --pdf_zoom_region must match number of clip_names')
    # Allow either one zoom window position or a number equal to number of clip names
    if args.pdf_zoom_window_position is not None:
        if len(args.pdf_zoom_window_position) == 1:
            args.pdf_zoom_window_position = [args.pdf_zoom_window_position[0] for _ in args.clip_names]
        elif len(args.pdf_zoom_window_position) != len(args.clip_names):
            raise ValueError('Number of positions specified by --pdf_zoom_window_position must either equal 1 or the '
                             'number of clip_names')

    if args.model_labels is None:
        args.model_labels = args.exp_names
    else:
        assert(len(args.model_labels) == len(args.exp_names))

    if not os.path.isdir(args.dest_path):
        os.makedirs(args.dest_path)

    if not args.no_pdf:
        if args.pdf_one_middle_frame:
            create_pdfs_one_middle_frame(args.clip_names, args.dataset_name, args.dest_path, args.exp_names,
                                         args.model_labels, args.results_root, args.frame_indexes,
                                         args.pdf_frame_width, args.pdf_frame_height, args.pdf_border_width,
                                         args.pdf_spacing, args.pdf_font_size, args.pdf_label_frame_width,
                                         args.pdf_zoom_region, args.pdf_zoom_window_position)
        else:
            create_pdfs(args.clip_names, args.dataset_name, args.dest_path, args.exp_names, args.model_labels,
                        args.results_root, args.frame_indexes, args.pdf_frame_width, args.pdf_frame_height,
                        args.pdf_border_width, args.pdf_spacing, args.pdf_font_size, args.pdf_label_frame_width,
                        args.pdf_zoom_region, args.pdf_zoom_window_position)

    if args.video_type is not None:
        pool = multiprocessing.Pool(processes=args.num_workers)
        create_video(args.clip_names, args.dataset_name, args.dest_path, args.exp_names, args.model_labels,
                     args.results_root, args.total_num_frames, args.video_type, pool)


if __name__ == '__main__':
    main()