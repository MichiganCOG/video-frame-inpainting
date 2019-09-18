import argparse
import os

import numpy as np
from scipy.stats.mstats import gmean

SUMMARY_METHOD_STR_TO_FN = {
    'mean': np.mean,
    'total': sum,
    'geometric_mean': gmean
}

def main():
    parser = argparse.ArgumentParser(description='Prints absolute difference in performance between two models on a '
                                                 'per-video basis. Positive differences mean that model 1 does '
                                                 'better, and negative differences mean that model 2 does better.')
    parser.add_argument('results_file_path_1', type=str,
                        help='The path to the results.npz file of the first model')
    parser.add_argument('results_file_path_2', type=str,
                        help='The path to the results.npz file of the second model')
    parser.add_argument('--metric', type=str, choices=['psnr', 'ssim'], default='ssim',
                        help='Metric used to compute frame-wise performance')
    parser.add_argument('--summary_method', type=str, choices=['mean', 'total', 'geometric_mean'], default='mean',
                        help='Summary statistic used to compute whole-video performance')
    args = parser.parse_args()

    # Extract result tables, which should contain the video list and the given metric
    tables_1 = np.load(args.results_file_path_1)
    tables_2 = np.load(args.results_file_path_2)

    # Make sure video lists contain the same videos (determined just on basename)
    video_list_1 = tables_1['video']
    video_list_2 = tables_2['video']
    assert(len(video_list_1) == len(video_list_2))
    sorted_video_basenames_1 = sorted(os.path.basename(x) for x in video_list_1)
    sorted_video_basenames_2 = sorted(os.path.basename(x) for x in video_list_2)
    assert(sorted_video_basenames_1 == sorted_video_basenames_2)

    # Get performance summary statistic on each video
    summary_fn = SUMMARY_METHOD_STR_TO_FN[args.summary_method]
    metric_summary_list_1 = summary_fn(tables_1[args.metric], axis=1)
    metric_summary_list_2 = summary_fn(tables_2[args.metric], axis=1)
    # Create dictionaries where keys are the video name and values are the video's performance summary statistic
    dict_1 = dict(zip([os.path.basename(x) for x in video_list_1], metric_summary_list_1))
    dict_2 = dict(zip([os.path.basename(x) for x in video_list_2], metric_summary_list_2))
    # Create another dict where the keys are the video name and values are the differences between model 1 and model 2
    dict_diffs = {}
    for k in dict_1:
        dict_diffs[k] = dict_1[k] - dict_2[k]

    # Sort the videos based on difference in performance
    sorted_tuples = sorted(dict_diffs.items(), key=lambda x: x[1])
    for video_name, diff in sorted_tuples:
        # print('%s\t%f' % (video_name, diff))
        print(video_name)

if __name__ == '__main__':
    main()