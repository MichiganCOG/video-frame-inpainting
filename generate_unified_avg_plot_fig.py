import argparse
import os
from pprint import pprint
from datetime import datetime

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.util.vis_utils import cm2in
from summarize_quant_results import draw_avg_error_on_ax

__SCRIPT_DIR__ = os.path.dirname(os.path.abspath(__file__))

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
    ax_psnr_T_a.set_position([.06, .25, .13, .68])
    ax_psnr_T_a.set_xlabel('Time step (m=%d)' % T_a)
    ax_psnr_T_a.set_ylabel('PSNR')
    ax_psnr_T_a.axis([1, T_a, psnr_range[0], psnr_range[1]])
    for i, model_label in enumerate(model_labels):
        try:
            psnr_table = np.load(quant_results_roots[0][i])['psnr']
        except IOError:
            raise ValueError('Failed to read file %s' % quant_results_roots[0][i])
        except Exception as e:
            raise e
        psnr_table[psnr_table > 100] = 100
        draw_avg_error_on_ax(ax_psnr_T_a, psnr_table, model_label)

    # Draw PSNR T=T_b plot
    ax_psnr_T_b = fig.add_subplot(111, label='b')
    ax_psnr_T_b.set_position([.22, .25, .18, .68])
    ax_psnr_T_b.set_xlabel('Time step (m=%d)' % T_b)
    ax_psnr_T_b.axis([1, T_a, psnr_range[0], psnr_range[1]])
    ax_psnr_T_b.tick_params(axis='y', labelleft=False)
    for i, model_label in enumerate(model_labels):
        try:
            psnr_table = np.load(quant_results_roots[1][i])['psnr']
        except IOError:
            raise ValueError('Failed to read file %s' % quant_results_roots[1][i])
        except Exception as e:
            raise e
        psnr_table[psnr_table > 100] = 100
        draw_avg_error_on_ax(ax_psnr_T_b, psnr_table, model_label)

    # Draw SSIM T=T_a plot
    ax_ssim_T_a = fig.add_subplot(111, label='c')
    ax_ssim_T_a.set_position([.5, .25, .13, .68])
    ax_ssim_T_a.set_xlabel('Time step (m=%d)' % T_a)
    ax_ssim_T_a.set_ylabel('SSIM')
    ax_ssim_T_a.axis([1, T_a, ssim_range[0], ssim_range[1]])
    for i, model_label in enumerate(model_labels):
        try:
            ssim_table = np.load(quant_results_roots[0][i])['ssim']
        except IOError:
            raise ValueError('Failed to read file %s' % quant_results_roots[0][i])
        except Exception as e:
            raise e
        draw_avg_error_on_ax(ax_ssim_T_a, ssim_table, model_label)

    # Draw SSIM T=T_b plot
    ax_ssim_T_b = fig.add_subplot(111, label='d')
    ax_ssim_T_b.set_position([.66, .25, .18, .68])
    ax_ssim_T_b.set_xlabel('Time step (m=%d)' % T_b)
    ax_ssim_T_b.axis([1, T_a, ssim_range[0], ssim_range[1]])
    ax_ssim_T_b.tick_params(axis='y', labelleft=False)
    for i, model_label in enumerate(model_labels):
        try:
            ssim_table = np.load(quant_results_roots[1][i])['ssim']
        except IOError:
            raise ValueError('Failed to read file %s' % quant_results_roots[1][i])
        except Exception as e:
            raise e
        draw_avg_error_on_ax(ax_ssim_T_b, ssim_table, model_label)

    # Draw legend to right of final plot
    ax_ssim_T_b.legend(loc='center', bbox_to_anchor=(1.46, .5))

    # Save the plot
    if not os.path.isdir(os.path.dirname(args.dest_path)):
        os.makedirs(os.path.dirname(args.dest_path))
    plt.savefig(args.dest_path)

if __name__ == '__main__':
    main()