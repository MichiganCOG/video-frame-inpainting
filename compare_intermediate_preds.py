from __future__ import division

import argparse
import os
from datetime import datetime
from warnings import warn

from src.util.vis_utils import in2cm, add_image_to_pdf, add_cropped_image_to_pdf, add_text_to_pdf, get_text_width, \
    create_pdf, ORANGE, PURPLE, YELLOW, GREEN, CYAN

__SCRIPT_DIR__ = os.path.abspath(os.path.dirname(__file__))


def generate_interp_net_pred_pdf(results_root, dataset_name, clip_names, exp_names, model_labels, dest_path, ts,
                                 frame_width_cm, frame_height_cm, border_width_cm, spacing_cm, font_size_pt,
                                 label_frame_width_cm, zoom_regions):
    max_text_width_cm = -1
    for model_label in model_labels:
        max_text_width_cm = max(max_text_width_cm, get_text_width(model_label, font_size_pt))

    # Compute height and width of final figures
    fig_width_cm = max_text_width_cm + label_frame_width_cm \
                   + 3 * (2 * border_width_cm + frame_width_cm + spacing_cm) - spacing_cm
    fig_height_cm = len(exp_names) * (2 * frame_height_cm + 4 * border_width_cm + 6 * spacing_cm) - 4 * spacing_cm \
                    + 2.11 * in2cm(font_size_pt / 72)

    bottom_labels = [('Before', 'adpt. conv.'), ('After', 'adpt. conv.'), ('Final', 'prediction')]

    for c, (clip_name, zoom_region) in enumerate(zip(clip_names, zoom_regions)):
        for t in ts:
            print('Creating PDF of size %f x %f cm' % (fig_width_cm, fig_height_cm))
            pdf = create_pdf(fig_width_cm, fig_height_cm, 'cm')

            for j, (text_0, text_1) in enumerate(bottom_labels):
                y_cm = len(exp_names) * (2 * frame_height_cm + 4 * border_width_cm + 6 * spacing_cm) - 3 * spacing_cm
                x_cm = (frame_width_cm + 2 * border_width_cm - get_text_width(text_0, font_size_pt)) / 2 \
                       + max_text_width_cm + label_frame_width_cm \
                       + j * (frame_width_cm + 2 * border_width_cm + spacing_cm)
                add_text_to_pdf(pdf, text_0, x_cm, y_cm, font_size_pt)
                y_cm = len(exp_names) * (2 * frame_height_cm + 4 * border_width_cm + 6 * spacing_cm) - 3 * spacing_cm \
                       + in2cm(font_size_pt / 72)
                x_cm = (frame_width_cm + 2 * border_width_cm - get_text_width(text_1, font_size_pt)) / 2 \
                       + max_text_width_cm + label_frame_width_cm \
                       + j * (frame_width_cm + 2 * border_width_cm + spacing_cm)
                add_text_to_pdf(pdf, text_1, x_cm, y_cm, font_size_pt)

            for i, (exp_name, model_label) in enumerate(zip(exp_names, model_labels)):
                # Draw the label for the current model
                y_offset = (2 * frame_height_cm + 4 * border_width_cm + spacing_cm - .7 * in2cm(font_size_pt / 72)) / 2
                y_cm = i * (2 * frame_height_cm + 4 * border_width_cm + 6 * spacing_cm) + y_offset
                cur_text_width_cm = get_text_width(model_label, font_size_pt)
                add_text_to_pdf(pdf, model_label, max_text_width_cm - cur_text_width_cm, y_cm, font_size_pt)

                images_root = os.path.join(results_root, dataset_name, 'images', exp_name, clip_name)
                if not os.path.isdir(images_root):
                    warn('Could not find image folder %s. Images will not be printed to video' % images_root)

                if zoom_region is None:
                    image_path = os.path.join(images_root, 'pred_middle_forward_%04d.png' % t)
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + 0 * (
                        frame_width_cm + 2 * border_width_cm + spacing_cm)
                    y_cm = border_width_cm + 2 * i * (frame_height_cm + 2 * border_width_cm + 3 * spacing_cm)
                    add_image_to_pdf(pdf, image_path, x_cm, y_cm, frame_width_cm, frame_height_cm, b_cm=border_width_cm,
                                     color=CYAN)

                    image_path = os.path.join(images_root, 'pred_middle_backward_%04d.png' % t)
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + 0 * (
                        frame_width_cm + 2 * border_width_cm + spacing_cm)
                    y_cm = border_width_cm + (2 * i + 1) * (frame_height_cm + 2 * border_width_cm + 3 * spacing_cm) - \
                           2 * spacing_cm
                    add_image_to_pdf(pdf, image_path, x_cm, y_cm, frame_width_cm, frame_height_cm, b_cm=border_width_cm,
                                     color=PURPLE)

                    image_path = os.path.join(images_root, 'interp_net_outputs_1_%04d.png' % t)
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + 1 * (
                        frame_width_cm + 2 * border_width_cm + spacing_cm)
                    y_cm = border_width_cm + 2 * i * (frame_height_cm + 2 * border_width_cm + 3 * spacing_cm)
                    add_image_to_pdf(pdf, image_path, x_cm, y_cm, frame_width_cm, frame_height_cm, b_cm=border_width_cm,
                                     color=CYAN)

                    image_path = os.path.join(images_root, 'interp_net_outputs_2_%04d.png' % t)
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + 1 * (
                        frame_width_cm + 2 * border_width_cm + spacing_cm)
                    y_cm = border_width_cm + (2 * i + 1) * (frame_height_cm + 2 * border_width_cm + 3 * spacing_cm) - \
                           2 * spacing_cm
                    add_image_to_pdf(pdf, image_path, x_cm, y_cm, frame_width_cm, frame_height_cm, b_cm=border_width_cm,
                                     color=PURPLE)

                    image_path = os.path.join(images_root, 'pred_middle_%04d.png' % t)
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + 2 * (
                        frame_width_cm + 2 * border_width_cm + spacing_cm)
                    y_cm = border_width_cm + (2 * i + 1) * (frame_height_cm + 2 * border_width_cm + 3 * spacing_cm) - \
                           2 * spacing_cm - (frame_height_cm + 2 * border_width_cm + spacing_cm) / 2
                    add_image_to_pdf(pdf, image_path, x_cm, y_cm, frame_width_cm, frame_height_cm, b_cm=border_width_cm,
                                     color=YELLOW)

                else:
                    image_path = os.path.join(images_root, 'pred_middle_forward_%04d.png' % t)
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + 0 * (
                    frame_width_cm + 2 * border_width_cm + spacing_cm)
                    y_cm = border_width_cm + 2 * i * (frame_height_cm + 2 * border_width_cm + 3 * spacing_cm)
                    add_cropped_image_to_pdf(pdf, image_path, zoom_region, x_cm, y_cm, frame_width_cm, frame_height_cm,
                                             b_cm=border_width_cm, color=CYAN)

                    image_path = os.path.join(images_root, 'pred_middle_backward_%04d.png' % t)
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + 0 * (
                    frame_width_cm + 2 * border_width_cm + spacing_cm)
                    y_cm = border_width_cm + (2 * i + 1) * (frame_height_cm + 2 * border_width_cm + 3 * spacing_cm) - \
                            2 * spacing_cm
                    add_cropped_image_to_pdf(pdf, image_path, zoom_region, x_cm, y_cm, frame_width_cm, frame_height_cm,
                                             b_cm=border_width_cm, color=PURPLE)

                    image_path = os.path.join(images_root, 'interp_net_outputs_1_%04d.png' % t)
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + 1 * (
                    frame_width_cm + 2 * border_width_cm + spacing_cm)
                    y_cm = border_width_cm + 2 * i * (frame_height_cm + 2 * border_width_cm + 3 * spacing_cm)
                    add_cropped_image_to_pdf(pdf, image_path, zoom_region, x_cm, y_cm, frame_width_cm, frame_height_cm,
                                             b_cm=border_width_cm, color=CYAN)

                    image_path = os.path.join(images_root, 'interp_net_outputs_2_%04d.png' % t)
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + 1 * (
                    frame_width_cm + 2 * border_width_cm + spacing_cm)
                    y_cm = border_width_cm + (2 * i + 1) * (frame_height_cm + 2 * border_width_cm + 3 * spacing_cm) - \
                           2 * spacing_cm
                    add_cropped_image_to_pdf(pdf, image_path, zoom_region, x_cm, y_cm, frame_width_cm, frame_height_cm,
                                             b_cm=border_width_cm, color=PURPLE)

                    image_path = os.path.join(images_root, 'pred_middle_%04d.png' % t)
                    x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + 2 * (
                        frame_width_cm + 2 * border_width_cm + spacing_cm)
                    y_cm = border_width_cm + (2 * i + 1) * (frame_height_cm + 2 * border_width_cm + 3 * spacing_cm) - \
                           2 * spacing_cm - (frame_height_cm + 2 * border_width_cm + spacing_cm) / 2
                    add_cropped_image_to_pdf(pdf, image_path, zoom_region, x_cm, y_cm, frame_width_cm, frame_height_cm,
                                             b_cm=border_width_cm, color=YELLOW)

            if not os.path.isdir(os.path.join(dest_path, clip_name)):
                os.makedirs(os.path.join(dest_path, clip_name))
            pdf.output(os.path.join(dest_path, clip_name, '%02d.pdf' % t))


def generate_bidirectional_pred_pdf(results_root, dataset_name, clip_names, exp_names, model_labels, dest_path, ts,
                                    frame_width_cm, frame_height_cm, border_width_cm, spacing_cm, font_size_pt,
                                    label_frame_width_cm, zoom_regions):
    max_text_width_cm = -1
    for model_label in model_labels:
        max_text_width_cm = max(max_text_width_cm, get_text_width(model_label, font_size_pt))

    # Compute height and width of final figures
    fig_width_cm = max_text_width_cm + label_frame_width_cm \
                   + 3 * (2 * border_width_cm + frame_width_cm + spacing_cm) - spacing_cm
    fig_height_cm = len(exp_names) * (frame_height_cm + 2 * border_width_cm + spacing_cm) + spacing_cm \
        + .7 * in2cm(font_size_pt / 72)

    image_name_templates = [
        'pred_middle_forward_%04d.png',
        'pred_middle_backward_%04d.png',
        'pred_middle_%04d.png'
    ]
    colors = [CYAN, PURPLE, YELLOW, GREEN]
    bottom_labels = ['Fwd', 'Bkwd', 'Final']

    for c, (zoom_region, clip_name) in enumerate(
            zip(zoom_regions, clip_names)):
        for t in ts:
            print('Creating PDF of size %f x %f cm' % (fig_width_cm, fig_height_cm))
            pdf = create_pdf(fig_width_cm, fig_height_cm, 'cm')

            y_cm = len(exp_names) * (frame_height_cm + 2 * border_width_cm + spacing_cm) + spacing_cm

            for j, text in enumerate(bottom_labels):
                x_cm = (frame_width_cm + 2 * border_width_cm - get_text_width(text, font_size_pt)) / 2 \
                    + max_text_width_cm + label_frame_width_cm \
                    + j * (frame_width_cm + 2 * border_width_cm + spacing_cm)
                add_text_to_pdf(pdf, text, x_cm, y_cm, font_size_pt)

            for i, (exp_name, model_label) in enumerate(zip(exp_names, model_labels)):
                # Draw the label for the current model
                y_offset = (frame_height_cm + 2 * border_width_cm - .7 * in2cm(font_size_pt / 72)) / 2
                y_cm = i * (frame_height_cm + 2 * border_width_cm + spacing_cm) + y_offset
                cur_text_width_cm = get_text_width(model_label, font_size_pt)
                add_text_to_pdf(pdf, model_label, max_text_width_cm - cur_text_width_cm, y_cm, font_size_pt)

                images_root = os.path.join(results_root, dataset_name, 'images', exp_name, clip_name)
                if not os.path.isdir(images_root):
                    warn('Could not find image folder %s. Images will not be printed to video' % images_root)

                if zoom_region is None:
                    # Draw the full frame
                    for j, template in enumerate(image_name_templates):
                        image_path = os.path.join(images_root, template % t)
                        x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + j * (
                            frame_width_cm + 2 * border_width_cm + spacing_cm)
                        y_cm = border_width_cm + i * (frame_height_cm + 2 * border_width_cm + spacing_cm)
                        add_image_to_pdf(pdf, image_path, x_cm, y_cm, frame_width_cm, frame_height_cm, b_cm=border_width_cm,
                                         color=colors[j])
                else:
                    # Draw the specified cropped region
                    for j, template in enumerate(image_name_templates):
                        image_path = os.path.join(images_root, template % t)
                        x_cm = max_text_width_cm + label_frame_width_cm + border_width_cm + j * (
                            frame_width_cm + 2 * border_width_cm + spacing_cm)
                        y_cm = border_width_cm + i * (frame_height_cm + 2 * border_width_cm + spacing_cm)
                        add_cropped_image_to_pdf(pdf, image_path, zoom_region, x_cm, y_cm, frame_width_cm,
                                                 frame_height_cm, b_cm=border_width_cm, color=colors[j])

            if not os.path.isdir(os.path.join(dest_path, clip_name)):
                os.makedirs(os.path.join(dest_path, clip_name))
            pdf.output(os.path.join(dest_path, clip_name, '%02d.pdf' % t))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_root', type=str, default='results',
                        help='Path to the "results" folder containing all experimental results')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the folder directly under "results" (e.g. "UCF-test_data_list_T=3")')
    parser.add_argument('--clip_names', type=str, nargs='+', required=True,
                        help='Names of the video clips to visualize (e.g. "v_Archery_g05_c04.avi_1-11")')
    parser.add_argument('--exp_names', type=str, nargs='+', required=True,
                        help='Folder names of one or more experiments (e.g. TAI_0200)')
    parser.add_argument('--model_labels', type=str, nargs='+', default=None,
                        help='Labels for the given experiments (e.g. "TAI")')
    parser.add_argument('--dest_path', type=str, default=os.path.join('visual_results', str(datetime.now())),
                        help='Path to place the generated PDFs')
    parser.add_argument('--ts', type=int, nargs='+', required=True, help='Time steps to visualize')
    parser.add_argument('--pdf_frame_width', type=float, default=1.73, help='Width of each video frame in cm')
    parser.add_argument('--pdf_frame_height', type=float, default=1.73, help='Height of each video frame in cm')
    parser.add_argument('--pdf_border_width', type=float, default=0.05,
                        help='Thickness of the border around each video frame in cm')
    parser.add_argument('--pdf_spacing', type=float, default=0.07,
                        help='Thickness of margin between each video frame in cm')
    parser.add_argument('--pdf_font_size', type=int, default=10, help='Font size for the PDF text')
    parser.add_argument('--pdf_label_frame_width', type=float, default=0.25,
                        help='Size of the margin between the labels and the video frames in cm')
    parser.add_argument('--pdf_zoom_region', type=float, nargs=4, action='append', default=None,
                        help='The patch to zoom in on defined as four decimal values specifying TL-x, TL-y, BR-x, '
                             'and BR-y coordinates (TL = top left, BR = bottom right). Use this flag multiple times '
                             'to specify a different region for each clip')

    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('bidirectional_pred')
    subparsers.add_parser('interp_net_pred')

    args = parser.parse_args()

    # Set zoom info to lists of None if not specified
    if args.pdf_zoom_region is None:
        args.pdf_zoom_region = [None for _ in args.clip_names]
    if args.pdf_zoom_region is not None and len(args.pdf_zoom_region) != len(args.clip_names):
        raise ValueError('Number of regions specified by --pdf_zoom_region must match number of clip_names')

    if args.model_labels is None:
        args.model_labels = args.exp_names
    else:
        assert(len(args.model_labels) == len(args.exp_names))

    if args.command == 'bidirectional_pred':
        generate_bidirectional_pred_pdf(args.results_root, args.dataset_name, args.clip_names, args.exp_names,
                                        args.model_labels, args.dest_path, args.ts, args.pdf_frame_width,
                                        args.pdf_frame_height, args.pdf_border_width, args.pdf_spacing,
                                        args.pdf_font_size, args.pdf_label_frame_width, args.pdf_zoom_region)
    elif args.command == 'interp_net_pred':
        generate_interp_net_pred_pdf(args.results_root, args.dataset_name, args.clip_names, args.exp_names,
                                     args.model_labels, args.dest_path, args.ts, args.pdf_frame_width,
                                     args.pdf_frame_height, args.pdf_border_width, args.pdf_spacing,
                                     args.pdf_font_size, args.pdf_label_frame_width, args.pdf_zoom_region)

if __name__ == '__main__':
    main()