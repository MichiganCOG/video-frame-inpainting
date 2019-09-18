import argparse
import os
from pprint import pprint
from datetime import datetime

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.util.vis_utils import cm2in

__SCRIPT_DIR__ = os.path.dirname(os.path.abspath(__file__))
LABEL_COLOR_MAP = {
    'Newson et al.': 'C3',
    'MCnet': 'C2',
    'Super SloMo': 'C1',
    'bi-TAI (ours)': 'C0'
}


def draw_video_perf_boxplot_on_ax(ax, error_table_list, labels, hide_labels=False):
    """

    :param ax: The PyPlot axis to draw on
    :param error_table_list: list of M N x T NumPy arrays
    :param labels: The labels associated with this data
    :param hide_labels: Whether to print the given labels on the y-axis
    """
    assert(len(error_table_list) == len(labels))

    # Define box and flier properties
    props = dict(
        boxprops=dict(linewidth=0.1),
        flierprops=dict(marker='|', markersize=4, markeredgecolor=(.9, .9, .9), markeredgewidth=0.1),
        whiskerprops=dict(linewidth=0.1),
        capprops=dict(linewidth=0.1),
        medianprops=dict(linewidth=0.1, color='black')
    )

    error_table_cat = np.stack(error_table_list)  # M x N x T
    # Compute the score for each video by taking the mean performance across all video frames
    video_scores_cat = error_table_cat.mean(axis=2)  # M x N
    # Reorder dimensions for boxplot call, and reverse order so first model is on top
    video_scores_cat = video_scores_cat.T[:, ::-1]  # N x M
    # Draw box plot with outliers (fliers)
    boxplot_items = ax.boxplot(video_scores_cat, vert=False, patch_artist=True, **props)
    # Add model labels in reverse order (so first one goes on top)
    ax.set_yticklabels('' if hide_labels else labels[::-1])
    # Colorize each box
    for i, patch in enumerate(boxplot_items['boxes'][::-1]):
        patch.set_facecolor(LABEL_COLOR_MAP[labels[i]] if labels[i] in LABEL_COLOR_MAP else 'C%d' % i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', type=str, default=os.path.join(__SCRIPT_DIR__, 'results'))
    parser.add_argument('--dest_path', type=str,
                        default=os.path.join(__SCRIPT_DIR__, 'summaries', str(datetime.now()), 'unified_avg_plot.pdf'))
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--T_a', type=int, required=True)
    parser.add_argument('--T_b', type=int, required=True)
    parser.add_argument('--exp_names', type=str, nargs='+', required=True)
    parser.add_argument('--model_labels', type=str, nargs='+', required=True)
    parser.add_argument('--psnr_range', type=float, nargs=2, required=True)
    parser.add_argument('--ssim_range', type=float, nargs=2, required=True)
    args = parser.parse_args()

    if len(args.exp_names) != len(args.model_labels):
        raise ValueError('Number of arguments to --exp_names and --model_labels must match')

    results_root = args.results_root
    dataset = args.dataset
    T_a = args.T_a
    T_b = args.T_b
    exp_names = args.exp_names
    model_labels = args.model_labels
    psnr_range = args.psnr_range
    ssim_range = args.ssim_range

    template = os.path.join(results_root, '{dataset}-test_data_list_T={T}', 'quantitative', '{exp_name}', 'results.npz')
    quant_results_roots = [
        [template.format(dataset=dataset, T=T_a, exp_name=exp_name) for exp_name in exp_names],
        [template.format(dataset=dataset, T=T_b, exp_name=exp_name) for exp_name in exp_names]
    ]

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 7
    fig = plt.figure(figsize=(cm2in(18.2), cm2in(4)))

    # Draw PSNR T=T_a plot
    ax_psnr_T_a = fig.add_subplot(111, label='a')
    ax_psnr_T_a.set_position([.12, .25, .2, .68])
    ax_psnr_T_a.set_xlabel('Mean PSNR (m=5)')
    ax_psnr_T_a.axis([psnr_range[0], psnr_range[1], 1, len(exp_names)])
    ax_psnr_T_a.tick_params(axis='y', left=False)
    psnr_tables_list = []
    for i, model_label in enumerate(model_labels):
        try:
            psnr_table = np.load(quant_results_roots[0][i])['psnr']
        except IOError:
            raise ValueError('Failed to read file %s' % quant_results_roots[0][i])
        except Exception as e:
            raise e
        psnr_tables_list.append(psnr_table)
    draw_video_perf_boxplot_on_ax(ax_psnr_T_a, psnr_tables_list, model_labels)

    # Draw PSNR T=T_b plot
    ax_psnr_T_b = fig.add_subplot(111, label='b')
    ax_psnr_T_b.set_position([.34, .25, .2, .68])
    ax_psnr_T_b.set_xlabel('Mean PSNR (m=10)')
    ax_psnr_T_b.axis([psnr_range[0], psnr_range[1], 1, len(exp_names)])
    ax_psnr_T_b.tick_params(axis='y', left=False)
    psnr_tables_list = []
    for i, model_label in enumerate(model_labels):
        try:
            psnr_table = np.load(quant_results_roots[1][i])['psnr']
        except IOError:
            raise ValueError('Failed to read file %s' % quant_results_roots[1][i])
        except Exception as e:
            raise e
        psnr_tables_list.append(psnr_table)
    draw_video_perf_boxplot_on_ax(ax_psnr_T_b, psnr_tables_list, model_labels, hide_labels=True)

    # Draw SSIM T=T_a plot
    ax_ssim_T_a = fig.add_subplot(111, label='c')
    ax_ssim_T_a.set_position([.56, .25, .2, .68])
    ax_ssim_T_a.set_xlabel('Mean SSIM (m=5)')
    ax_ssim_T_a.axis([ssim_range[0], ssim_range[1], 1, len(exp_names)])
    ax_ssim_T_a.tick_params(axis='y', left=False)
    ssim_tables_list = []
    for i, model_label in enumerate(model_labels):
        try:
            ssim_table = np.load(quant_results_roots[0][i])['ssim']
        except IOError:
            raise ValueError('Failed to read file %s' % quant_results_roots[0][i])
        except Exception as e:
            raise e
        ssim_tables_list.append(ssim_table)
    draw_video_perf_boxplot_on_ax(ax_ssim_T_a, ssim_tables_list, model_labels, hide_labels=True)

    # Draw SSIM T=T_b plot
    ax_ssim_T_b = fig.add_subplot(111, label='d')
    ax_ssim_T_b.set_position([.78, .25, .2, .68])
    ax_ssim_T_b.set_xlabel('Mean SSIM (m=10)')
    ax_ssim_T_b.axis([ssim_range[0], ssim_range[1], 1, len(exp_names)])
    ax_ssim_T_b.tick_params(axis='y', left=False)
    ssim_tables_list = []
    for i, model_label in enumerate(model_labels):
        try:
            ssim_table = np.load(quant_results_roots[1][i])['ssim']
        except IOError:
            raise ValueError('Failed to read file %s' % quant_results_roots[1][i])
        except Exception as e:
            raise e
        ssim_tables_list.append(ssim_table)
    draw_video_perf_boxplot_on_ax(ax_ssim_T_b, ssim_tables_list, model_labels, hide_labels=True)

    plt.savefig(args.dest_path)

if __name__ == '__main__':
    main()