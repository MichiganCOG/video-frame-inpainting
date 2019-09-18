#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: compute_summarize_quant_results.sh QUAL_RESULT_ROOT QUANT_RESULT_ROOT K T"
    exit 1
fi

QUAL_RESULT_ROOT="$1"
QUANT_RESULT_ROOT="$2"
K="$3"
T="$4"

python compute_quant_results.py "$QUAL_RESULT_ROOT" "$QUANT_RESULT_ROOT" "$K" "$T"

python summarize_quant_results.py avg_metric_plot \
    --quant_results_roots "$QUANT_RESULT_ROOT" \
    --T "$T" \
    --metric "psnr" \
    --range 24 38 \
    --dest_path "$QUANT_RESULT_ROOT"

python summarize_quant_results.py avg_metric_plot \
    --quant_results_roots "$QUANT_RESULT_ROOT" \
    --T "$T" \
    --metric "ssim" \
    --range .75 .98 \
    --dest_path "$QUANT_RESULT_ROOT"

python summarize_quant_results.py video_metric_text_file \
    --quant_results_roots "$QUANT_RESULT_ROOT" \
    --metric "psnr" \
    --dest_path "$QUANT_RESULT_ROOT"

python summarize_quant_results.py video_metric_text_file \
    --quant_results_roots "$QUANT_RESULT_ROOT" \
    --metric "ssim" \
    --dest_path "$QUANT_RESULT_ROOT"

python summarize_quant_results.py metric_summary_text_file \
    --quant_results_roots "$QUANT_RESULT_ROOT" \
    --metric "psnr" \
    --mean_precision 2 \
    --std_err_precision 3 \
    --dest_path "$QUANT_RESULT_ROOT"

python summarize_quant_results.py metric_summary_text_file \
    --quant_results_roots "$QUANT_RESULT_ROOT" \
    --metric "ssim" \
    --mean_precision 4 \
    --std_err_precision 6 \
    --dest_path "$QUANT_RESULT_ROOT"