#!/bin/bash

### GENERATE QUANTITATIVE PLOTS ###

mkdir -p paper_figs

python generate_unified_avg_plot_fig.py \
    --results_root results \
    --dest_path paper_figs/kth_avg_plot.pdf \
    --dataset KTH \
    --T_a 5 \
    --T_b 10 \
    --exp_names "bi-TAI" "SuperSloMo" "MCnet" "Newson" "TW_P_F" \
    --model_labels "bi-TAI (ours)" "Super SloMo" "MCnet" "Newson et al." "TW_P_F" \
    --psnr_range 25 38 \
    --ssim_range .82 .98

python generate_unified_avg_plot_fig.py \
    --results_root results \
    --dest_path paper_figs/ucf_avg_plot.pdf \
    --dataset UCF \
    --T_a 3 \
    --T_b 5 \
    --exp_names "bi-TAI" "SuperSloMo_val_test" "MCnet" "Newson" "TW_P_F" \
    --model_labels "bi-TAI (ours)" "Super SloMo" "MCnet" "Newson et al." "TW_P_F" \
    --psnr_range 24 32 \
    --ssim_range .79 .92

python generate_unified_avg_plot_fig.py \
    --results_root results \
    --dest_path paper_figs/hmdb_avg_plot.pdf \
    --dataset HMDB \
    --T_a 3 \
    --T_b 5 \
    --exp_names "bi-TAI" "SuperSloMo_val_test" "MCnet" "Newson" "TW_P_F" \
    --model_labels "bi-TAI (ours)" "Super SloMo" "MCnet" "Newson et al." "TW_P_F" \
    --psnr_range 24 32 \
    --ssim_range .75 .9

python generate_unified_boxplot_fig.py \
    --results_root results \
    --dest_path paper_figs/kth_boxplot.pdf \
    --dataset KTH \
    --T_a 5 \
    --T_b 10 \
    --exp_names "Newson" "MCnet" "SuperSloMo" "bi-TAI" \
    --model_labels "Newson et al." "MCnet" "Super SloMo" "bi-TAI (ours)" \
    --psnr_range 20 44 \
    --ssim_range .75 1

python generate_unified_avg_plot_fig.py \
    --results_root results \
    --dest_path paper_figs/imagenet_avg_plot.pdf \
    --dataset Imagenet \
    --T_a 3 \
    --T_b 5 \
    --exp_names "bi-TAI" "SuperSloMo_val_test" "MCnet" "Newson" "TW_P_F" \
    --model_labels "bi-TAI (ours)" "Super SloMo" "MCnet" "Newson et al." "TW_P_F" \
    --psnr_range 23 30 \
    --ssim_range .68 .84


### GENERATE PERFORMANCE TABLES ###

function generate_metric_summary_text_file {
    QUANT_RESULTS_ROOT=$1
    local -n EXP_NAMES_L=$2
    local -n LABELS_L=$3
    PSNR_TABLE_PATH=$4
    SSIM_TABLE_PATH=$5

    echo "Creating PSNR and SSIM tables from $QUANT_RESULTS_ROOT..."

    TEMP_FIG_ROOT="/tmp/$RANDOM"
    mkdir -p "$TEMP_FIG_ROOT"

    python summarize_quant_results.py metric_summary_text_file \
        --quant_results_roots "${EXP_NAMES_L[@]/#/$QUANT_RESULTS_ROOT/}" \
        --labels "${LABELS_L[@]}" \
        --dest_path "$TEMP_FIG_ROOT" \
        --metric psnr \
        --mean_precision 2 \
        --std_err_precision 3

    python summarize_quant_results.py metric_summary_text_file \
        --quant_results_roots "${EXP_NAMES_L[@]/#/$QUANT_RESULTS_ROOT/}" \
        --labels "${LABELS_L[@]}" \
        --dest_path "$TEMP_FIG_ROOT" \
        --metric ssim \
        --mean_precision 4 \
        --std_err_precision 6

    cp "$TEMP_FIG_ROOT/psnr_perf_summary.txt" "$PSNR_TABLE_PATH"
    cp "$TEMP_FIG_ROOT/ssim_perf_summary.txt" "$SSIM_TABLE_PATH"

    rm -r "$TEMP_FIG_ROOT"
}

mkdir -p quant_tables

## kth_baselines
EXP_NAMES=( TW_P_F Newson MCnet SuperSloMo bi-TAI )
LABELS=( "TW_P_F" "Newson et al." "MCnet" "Super SloMo" "bi-TAI (ours)" )
generate_metric_summary_text_file \
    results/KTH-test_data_list_T=5/quantitative \
    EXP_NAMES LABELS "quant_tables/kth_baselines_m=5_psnr.txt" "quant_tables/kth_baselines_m=5_ssim.txt"
generate_metric_summary_text_file \
    results/KTH-test_data_list_T=10/quantitative \
    EXP_NAMES LABELS "quant_tables/kth_baselines_m=10_psnr.txt" "quant_tables/kth_baselines_m=10_ssim.txt"

# kth_ablation
EXP_NAMES=( bi-SA bi-TWA bi-TWI bi-TAI )
LABELS=( "bi-SA" "bi-TWA" "bi-TWI" "bi-TAI (full)" )
generate_metric_summary_text_file \
    results/KTH-test_data_list_T=5/quantitative \
    EXP_NAMES LABELS "quant_tables/kth_ablation_m=5_psnr.txt" "quant_tables/kth_ablation_m=5_ssim.txt"
generate_metric_summary_text_file \
    results/KTH-test_data_list_T=10/quantitative \
    EXP_NAMES LABELS "quant_tables/kth_ablation_m=10_psnr.txt" "quant_tables/kth_ablation_m=10_ssim.txt"

# ucf_hmdb (UCF-101 tables)
EXP_NAMES=( TW_P_F Newson MCnet SuperSloMo_val_test bi-TAI )
LABELS=( "TW_P_F" "Newson et al." "MCnet" "Super SloMo" "bi-TAI (ours)" )
generate_metric_summary_text_file \
    results/UCF-test_data_list_T=3/quantitative \
    EXP_NAMES LABELS "quant_tables/ucf_baselines_m=3_psnr.txt" "quant_tables/ucf_baselines_m=3_ssim.txt"
generate_metric_summary_text_file \
    results/UCF-test_data_list_T=5/quantitative \
    EXP_NAMES LABELS "quant_tables/ucf_baselines_m=5_psnr.txt" "quant_tables/ucf_baselines_m=5_ssim.txt"

# ucf_hmdb (HMDB-51 tables)
EXP_NAMES=( TW_P_F Newson MCnet SuperSloMo_val_test bi-TAI )
LABELS=( "TW_P_F" "Newson et al." "MCnet" "Super SloMo" "bi-TAI (ours)" )
generate_metric_summary_text_file \
    results/HMDB-test_data_list_T=3/quantitative \
    EXP_NAMES LABELS "quant_tables/hmdb_baselines_m=3_psnr.txt" "quant_tables/hmdb_baselines_m=3_ssim.txt"
generate_metric_summary_text_file \
    results/HMDB-test_data_list_T=5/quantitative \
    EXP_NAMES LABELS "quant_tables/hmdb_baselines_m=5_psnr.txt" "quant_tables/hmdb_baselines_m=5_ssim.txt"

# imagenet (Imagenet-VID tables)
EXP_NAMES=( TW_P_F Newson MCnet SuperSloMo_val_test bi-TAI )
LABELS=( "TW_P_F" "Newson et al." "MCnet" "Super SloMo" "bi-TAI (ours)" )
generate_metric_summary_text_file \
    results/Imagenet-test_data_list_T=3/quantitative \
    EXP_NAMES LABELS "quant_tables/imagenet_baselines_m=3_psnr.txt" "quant_tables/imagenet_baselines_m=3_ssim.txt"
generate_metric_summary_text_file \
    results/Imagenet-test_data_list_T=5/quantitative \
    EXP_NAMES LABELS "quant_tables/imagenet_baselines_m=5_psnr.txt" "quant_tables/imagenet_baselines_m=5_ssim.txt"
