import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import prettytable
from scipy.stats.mstats import gmean

from src.util.vis_utils import cm2in

__SUBPARSER_ARGS__ = {
    'quant_results_roots': dict(type=str, nargs='+', required=True, help='Paths where each results.npz file is stored'),
    'labels': dict(type=str, nargs='+', default=None),
    'dest_path': dict(type=str, required=True, help='Folder to save the plots to'),
    'metric': dict(type=str, choices=['psnr', 'ssim'], required=True, help='The metric to use for the plot or summary'),
    'T': dict(type=int, required=True, help='Number of middle frames'),
    'range': dict(type=float, nargs=2, required=True, help='Range of possible values for the plot'),
    'fig_size': dict(type=float, nargs=2, default=None, help='Matplotlib figure size for the plots in cm'),
    'title': dict(type=str, default='', help='Title for all plots'),
    'summary_method': dict(type=str, default='mean', help='Method to compute summary value for each video'),
    'mean_precision': dict(type=int, default=4, help='Number of decimal places to keep in the reported mean'),
    'std_err_precision': dict(type=int, default=4, help='Number of decimal places to keep in the reported std. err.')
}


def draw_avg_error_on_ax(ax, error_table, label):
    """Draws an average PSNR or SSIM error plot and either saves it to disk or returns the image as a np.ndarray.

    :param ax: The figure axis to plot on
    :param error_table: The error values to plot as a N x T np.ndarray
    """

    N, T = error_table.shape
    x = np.arange(1, T+1)
    # Compute average scores and their standard errors
    avg_scores = np.mean(error_table, axis=0)
    std_errs = np.std(error_table, axis=0) / np.sqrt(N)
    # Add grid to background
    ax.grid(True, linewidth=0.1, color=(0.9, 0.9, 0.9, 1))
    # Plot the scores
    ax.plot(x, avg_scores, label=label, linewidth=0.8)
    # Fill in within two standard errors of the scores (covers 95% of variation)
    ax.fill_between(x, avg_scores - 2*std_errs, avg_scores + 2*std_errs, alpha=0.2)
    # Set line thickness
    [i.set_linewidth(0.1) for i in ax.spines.itervalues()]
    ax.tick_params(width=0.1)
    # Make ticks only occur at integer frame indexes
    ax.set_xticks(np.arange(1, T+1))


def prepare_figure_and_axis(x_label, y_label, lims, fig_size, title, use_legend_padding=True):
    """Create a figure with a single axis for plotting on it.

    :param x_label: The text along the x-axis
    :param y_label: The text along the y-axis
    :param lims: The range of values displayed in the plot
    :param fig_size: Tuple describing the size of the figure
    :param title: The title text for the plot
    :return: One Figure and one Axis object
    """

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    # Set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Set look
    ax.axis(lims)
    if use_legend_padding:
        ax.set_position([.3, .39, .65, .5])
    else:
        ax.set_position([.2, .15, .75, .77])

    return fig, ax


def generate_video_scores(video_list, error_table, summary_method):
    ''' Generate summary values for each video.

    Parameters:
        video_list: the names for each video
        error_table: a matrix that consists of lists of errors for each video
        summary_method: the summary method chosen for each video. You can set
            in the command as 'mean', 'total' and 'geometric_mean'

    Returns:
        a list with pairs of video name and summary value
    '''
    summary_score = {}
    for video_name, video_errors in zip(video_list, error_table):
        if summary_method == 'mean': score = np.mean(video_errors)
        if summary_method == 'total': score = sum(video_errors)
        if summary_method == 'geometric_mean': score = gmean(video_errors)
        summary_score[video_name] = score
    sorted_summary_value_video_pairs = sorted(summary_score.items(), key=lambda x: x[1])
    return sorted_summary_value_video_pairs


def plot_summary_scores(ax, summary_score, label, summary_method, lims):
    '''Plot the summary score for videos.

    Parameters:
        summary_score: a list of tuples of video name and summary score
    '''
    x = np.arange(1, len(summary_score) + 1)
    scores = [item[1] for item in summary_score]
    ax.plot(x, scores, label=label, marker='')


def boxplot_summary_scores(ax, df, xlabel, lims=None, title=None):
    '''Boxplot the summary score for videos.

    Parameters:
        ax: the axis to plot
        df: data frame for score
        lims: the range of values displayed in the plot
        title: the title text for the plot
    '''
    ax.axis(lims)
    ax.set_position([.15, .245, .78, .6])
    sns.boxplot(ax=ax, x=df[xlabel], y=df["labels"], showfliers=False, linewidth=0.1)
    [i.set_linewidth(0.1) for i in ax.spines.itervalues()]
    ax.tick_params(width=0.1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylabel('')


def finalize_figure_and_axis(fig, ax):
    ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(.31, -.33))


def create_avg_metric_plot(T, all_tables, avg_metric_y_lims, dest_path, fig_size, result_root_label_pairs, title, metric):
    if fig_size is None:
        fig_size = (cm2in(4.36), cm2in(5))
    else:
        fig_size = tuple([cm2in(t) for t in fig_size])
    plt.rcParams['font.size'] = 5.5

    metric_tables = all_tables[metric]
    avg_metric_plot_lims = [1, T, avg_metric_y_lims[0], avg_metric_y_lims[1]]
    fig, ax = prepare_figure_and_axis('time steps', metric.upper(), avg_metric_plot_lims, fig_size, title)
    for quant_results_root, label in result_root_label_pairs:
        metric_table = metric_tables[quant_results_root]
        draw_avg_error_on_ax(ax, metric_table, label)
    finalize_figure_and_axis(fig, ax)
    fig.savefig(os.path.join(dest_path, '%s_final.pdf' % metric))
    fig.savefig(os.path.join(dest_path, '%s_final.png' % metric))


def create_video_metric_text_file(all_tables, dest_path, result_root_label_pairs, summary_method, metric):
    metric_score_output_path = os.path.join(dest_path, 'sorted_%s_scores.txt' % metric)
    metric_score_output_file = open(metric_score_output_path, 'w')
    for quant_results_root, label in result_root_label_pairs:
        metric_table = all_tables[metric][quant_results_root]
        video_list = all_tables['videos'][quant_results_root]
        metric_score_video_pairs = generate_video_scores(video_list, metric_table, summary_method)
        if label is not None:
            metric_score_output_file.write('%s:\n' % label)
        for pair in metric_score_video_pairs:
            metric_score_output_file.write("%s, %s\n" % pair)
        metric_score_output_file.write('\n')
    metric_score_output_file.close()


def create_sorted_metric_plot(all_tables, dest_path, fig_size, result_root_label_pairs, sorted_metric_y_lims,
                            summary_method, title, metric):
    num_videos = all_tables[metric].values()[0].shape[0]
    sorted_metric_plot_lims = [-10, num_videos + 10, sorted_metric_y_lims[0], sorted_metric_y_lims[1]]
    fig, ax = prepare_figure_and_axis('Video rank', '%s score (%s)' % (metric.upper(), summary_method),
                                      sorted_metric_plot_lims, fig_size, title)
    for quant_results_root, label in result_root_label_pairs:
        metric_table = all_tables[metric][quant_results_root]
        video_list = all_tables['videos'][quant_results_root]
        metric_score_video_pairs = generate_video_scores(video_list, metric_table, summary_method)
        plot_summary_scores(ax, metric_score_video_pairs, label, summary_method, sorted_metric_plot_lims)
    finalize_figure_and_axis(fig, ax)
    fig.savefig(os.path.join(dest_path, 'sorted_%s_scores.png' % metric))
    fig.savefig(os.path.join(dest_path, 'sorted_%s_scores.pdf' % metric))


def create_metric_box_plot(all_tables, box_metric_y_lims, dest_path, fig_size, result_root_label_pairs, summary_method,
                           title, metric):
    if fig_size is None:
        fig_size = (cm2in(4.3), cm2in(4))
    else:
        fig_size = tuple([cm2in(t) for t in fig_size])
    plt.rcParams['font.size'] = 6

    box_metric_plot_lims = [box_metric_y_lims[0], box_metric_y_lims[1], 0, 1]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    all_scores = []
    box_plot_labels = []
    for quant_results_root, label in result_root_label_pairs:
        metric_table = all_tables[metric][quant_results_root]
        video_list = all_tables['videos'][quant_results_root]
        metric_score_video_pairs = generate_video_scores(video_list, metric_table, summary_method)
        score = [item[1] for item in metric_score_video_pairs]
        for s in score:
            all_scores.append(s)
            box_plot_labels.append(label)
    xlabel= '%s (%s)' % (metric.upper(), summary_method)
    df = pd.DataFrame(data={"labels": box_plot_labels, xlabel: all_scores})
    boxplot_summary_scores(ax, df, xlabel, box_metric_plot_lims, title)
    fig.savefig(os.path.join(dest_path, 'boxplot_%s_scores.png' % metric))
    fig.savefig(os.path.join(dest_path, 'boxplot_%s_scores.pdf' % metric))


def create_metric_summary_text_file(all_tables, dest_path, result_root_label_pairs, metric, mean_precision,
                                    std_err_precision):
    metric_perf_summary_output_path = os.path.join(dest_path, '%s_perf_summary.txt' % metric)
    # metric_perf_summary_output_file = open(metric_perf_summary_output_path, 'w')
    table = prettytable.PrettyTable(['Model', 'Mean', 'StdErr'])
    for quant_results_root, label in result_root_label_pairs:
        metric_table = all_tables[metric][quant_results_root]
        # Get per-video scores by taking mean score across all video frames
        per_video_scores = metric_table.mean(axis=1)
        # Get mean and standard error of per-video scores
        mean = per_video_scores.mean()
        std_err = per_video_scores.std() / np.sqrt(per_video_scores.size)
        # if label is not None:
        #     metric_perf_summary_output_file.write('%s: ' % label)
        # metric_perf_summary_output_file.write('%f +- %f\n' % (mean, std_err))
        table.add_row([label, '%.{}f'.format(mean_precision) % mean, '%.{}f'.format(std_err_precision) % std_err])
    with open(metric_perf_summary_output_path, 'w') as f:
        f.write(str(table))
    # metric_perf_summary_output_file.close()


def add_args_to_subparser(subparser, *arg_keys):
    for arg_key in arg_keys:
        subparser.add_argument('--%s' % arg_key, **__SUBPARSER_ARGS__[arg_key])


def main():

    ### CONSTRUCT ARGUMENT PARSER WITH SUBCOMMANDS ###

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    camp_parser = subparsers.add_parser('avg_metric_plot')
    add_args_to_subparser(camp_parser, 'quant_results_roots', 'labels', 'T', 'range', 'fig_size', 'dest_path',
                          'title', 'metric')

    cvmtf_parser = subparsers.add_parser('video_metric_text_file')
    add_args_to_subparser(cvmtf_parser, 'quant_results_roots', 'labels', 'dest_path', 'metric', 'summary_method')

    csmp_parser = subparsers.add_parser('sorted_metric_plot')
    add_args_to_subparser(csmp_parser, 'quant_results_roots', 'labels', 'dest_path', 'metric', 'range', 'fig_size',
                          'title', 'summary_method')

    cmbp_parser = subparsers.add_parser('metric_box_plot')
    add_args_to_subparser(cmbp_parser, 'quant_results_roots', 'labels', 'dest_path', 'metric', 'range', 'fig_size',
                          'title', 'summary_method')

    cmstf_parser = subparsers.add_parser('metric_summary_text_file')
    add_args_to_subparser(cmstf_parser, 'quant_results_roots', 'labels', 'dest_path', 'metric', 'mean_precision',
                          'std_err_precision')

    args, _ = parser.parse_known_args()

    ### PERFORM COMPLEX OPERATIONS SHARED ACROSS ALL COMMANDS ###

    # Set font to Times
    plt.rcParams['font.family'] = 'serif'
    # Associate labels for each model with the corresponding results folder
    if args.labels is None:
        args.labels = [None for _ in (args.quant_results_roots)]
    else:
        assert (len(args.labels) == len(args.quant_results_roots))
    result_root_label_pairs = zip(args.quant_results_roots, args.labels)
    if not os.path.isdir(args.dest_path):
        os.makedirs(args.dest_path)

    # Extract PSNR and SSIM tables from results file
    all_tables = {'psnr': {}, 'ssim': {}, "videos": {}}
    for quant_results_root in args.quant_results_roots:
        tables = np.load(os.path.join(quant_results_root, 'results.npz'))
        all_tables['psnr'][quant_results_root] = tables['psnr']
        all_tables['psnr'][quant_results_root][tables['psnr'] > 100] = 100
        all_tables['ssim'][quant_results_root] = tables['ssim']
        all_tables['videos'][quant_results_root] = tables['video']

    ### ASSIGN FUNCTION CALLS TO EACH COMMAND ###

    if args.command == 'avg_metric_plot':
        create_avg_metric_plot(args.T, all_tables, args.range, args.dest_path, args.fig_size, result_root_label_pairs,
                               args.title, args.metric)
    if args.command == 'video_metric_text_file':
        create_video_metric_text_file(all_tables, args.dest_path, result_root_label_pairs, args.summary_method,
                                      args.metric)
    if args.command == 'sorted_metric_plot':
        create_sorted_metric_plot(all_tables, args.dest_path, args.fig_size, result_root_label_pairs, args.range,
                                  args.summary_method, args.title, args.metric)
    if args.command == 'metric_box_plot':
        create_metric_box_plot(all_tables, args.range, args.dest_path, args.fig_size, result_root_label_pairs,
                               args.summary_method, args.title, args.metric)
    if args.command == 'metric_summary_text_file':
        create_metric_summary_text_file(all_tables, args.dest_path, result_root_label_pairs, args.metric,
                                        args.mean_precision, args.std_err_precision)


if __name__ == '__main__':
    main()
