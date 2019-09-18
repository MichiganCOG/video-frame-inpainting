#!/bin/bash

### MAKE TEMPORARY DIRECTORY WITH ALL FIGURES ###

TEMP_FIG_ROOT="/tmp/$RANDOM"
mkdir -p "$TEMP_FIG_ROOT"

### QUALITATIVE COMPARISON OF FINAL PREDICTIONS ###

## kth_qual_good_a
python compare_visual_results.py \
    --results_root results \
    --dataset_name KTH-test_data_list_T=5 \
    --frame_indexes 1 3 5 7 9 11 13 \
    --clip_names \
        person17_handwaving_d1_uncomp.avi_397-411 \
    --exp_names "MCnet" "Newson" "SuperSloMo" "bi-TAI" \
    --model_labels "MCnet" "Newson et al." "Super SloMo" "bi-TAI (ours)" \
    --dest_path "$TEMP_FIG_ROOT/KTH_T=5_good" \
    --pdf_frame_width 1.65 \
    --pdf_frame_height 1.65 \
    --pdf_spacing 0.07

# kth_qual_good_b
python compare_visual_results.py \
    --results_root results \
    --dataset_name KTH-test_data_list_T=5 \
    --frame_indexes 5 7 9 \
    --clip_names \
        person18_boxing_d2_uncomp.avi_115-129 \
        person19_handclapping_d2_uncomp.avi_280-294 \
        person20_handwaving_d2_uncomp.avi_179-193 \
        person25_running_d4_uncomp.avi_263-277 \
    --exp_names "bi-TAI" \
    --model_labels "bi-TAI (ours)" \
    --dest_path "$TEMP_FIG_ROOT/KTH_T=5_good" \
    --pdf_frame_width 1.65 \
    --pdf_frame_height 1.65 \
    --pdf_spacing 0.07

# kth_qual_bad
python compare_visual_results.py \
    --results_root results \
    --dataset_name KTH-test_data_list_T=5 \
    --frame_indexes 5 7 9 \
    --clip_names \
        person22_boxing_d2_uncomp.avi_31-45 \
    --pdf_zoom_region .48 .1 .68 .3 \
    --pdf_zoom_window_position .03 .55 .45 .97 \
    --exp_names "SuperSloMo" "bi-TAI" \
    --model_labels "Super SloMo" "bi-TAI (ours)" \
    --dest_path "$TEMP_FIG_ROOT/KTH_T=5_bad" \
    --pdf_frame_width 1.65 \
    --pdf_frame_height 1.65 \
    --pdf_spacing 0.07

# ucf_qual_good
python compare_visual_results.py \
    --results_root results \
    --dataset_name UCF-test_data_list_T=3 \
    --frame_indexes 3 5 7 \
    --clip_names \
        v_FrisbeeCatch_g02_c03.avi_1-11 \
        v_LongJump_g06_c04.avi_1-11 \
    --pdf_zoom_region .7 .38 .9 .78 \
    --pdf_zoom_window_position 0 0 .5 1 \
    --pdf_zoom_region .55 .35 .7 .65 \
    --pdf_zoom_window_position 0 0 .5 1 \
    --exp_names "Newson" "SuperSloMo_val_test" "bi-TAI" \
    --model_labels "Newson et al." "Super SloMo" "bi-TAI (ours)" \
    --dest_path "$TEMP_FIG_ROOT/UCF_T=3_good" \
    --pdf_frame_width 2.15 \
    --pdf_frame_height 1.6125 \
    --pdf_spacing 0.07 \
    --pdf_one_middle_frame

# ucf_qual_bad
python compare_visual_results.py \
    --results_root results \
    --dataset_name UCF-test_data_list_T=3 \
    --frame_indexes 3 5 7 \
    --clip_names \
        v_Biking_g04_c05.avi_1-11 \
    --pdf_zoom_region 0.25 0.15 0.5 0.5 \
    --pdf_zoom_window_position 0 0 0.714 1 \
    --exp_names "Newson" "SuperSloMo_val_test" "bi-TAI" \
    --model_labels "Newson et al." "Super SloMo" "bi-TAI (ours)" \
    --dest_path "$TEMP_FIG_ROOT/UCF_T=3_bad" \
    --pdf_frame_width 2.15 \
    --pdf_frame_height 1.6125 \
    --pdf_spacing 0.07 \
    --pdf_one_middle_frame

# hmdb_qual_good
python compare_visual_results.py \
    --results_root results \
    --dataset_name HMDB-test_data_list_T=3 \
    --frame_indexes 3 5 7 \
    --clip_names \
        St__Louis_Goalkeeping__Academy_elite_training_jump_f_nm_np1_ri_bad_9.avi_1-11 \
        5_Min_Tone_Abs_Workout_2__Fitness_Training_w__Tammy_situp_f_nm_np1_le_goo_5.avi_1-11 \
    --pdf_zoom_region .5 .49 .66 .7 \
    --pdf_zoom_window_position 0 0 .762 1 \
    --pdf_zoom_region .77 .4 .97 .8 \
    --pdf_zoom_window_position 0 0 .5 1 \
    --exp_names "Newson" "SuperSloMo_val_test" "bi-TAI" \
    --model_labels "Newson et al." "Super SloMo" "bi-TAI (ours)" \
    --dest_path "$TEMP_FIG_ROOT/HMDB_T=3_good" \
    --pdf_frame_width 2.15 \
    --pdf_frame_height 1.6125 \
    --pdf_spacing 0.07 \
    --pdf_one_middle_frame

# hmdb_qual_bad
python compare_visual_results.py \
    --results_root results \
    --dataset_name HMDB-test_data_list_T=3 \
    --frame_indexes 3 4 5 6 7 \
    --clip_names \
        AboutABoy_throw_f_nm_np1_ba_med_2.avi_1-11 \
    --exp_names "Newson" "SuperSloMo_val_test" "bi-TAI" \
    --model_labels "Newson et al." "Super SloMo" "bi-TAI (ours)" \
    --dest_path "$TEMP_FIG_ROOT/HMDB_T=3_bad" \
    --pdf_frame_width 2.15 \
    --pdf_frame_height 1.6125 \
    --pdf_spacing 0.07

# imagenet_qual_good
python compare_visual_results.py \
    --results_root results \
    --dataset_name Imagenet-test_data_list_T=3 \
    --frame_indexes 3 5 7 \
    --clip_names \
        ILSVRC2015_test_00027005.mkv_1-11 \
        ILSVRC2015_test_00199002.mkv_1-11 \
    --pdf_zoom_region .55 .18 .95 .58 \
    --pdf_zoom_window_position 0 0 .5 .5 \
    --pdf_zoom_region .45 .45 .95 .85 \
    --pdf_zoom_window_position 0 0 .5 .4 \
    --exp_names "Newson" "SuperSloMo_val_test" "bi-TAI" \
    --model_labels "Newson et al." "Super SloMo" "bi-TAI (ours)" \
    --dest_path "$TEMP_FIG_ROOT/Imagenet_T=3_good" \
    --pdf_frame_width 2.15 \
    --pdf_frame_height 1.6125 \
    --pdf_spacing 0.07 \
    --pdf_one_middle_frame

### COMPARISON OF INTERMEDIATE PREDICTIONS ###

python compare_visual_results.py \
    --results_root results \
    --dataset_name KTH-test_data_list_T=5 \
    --frame_indexes 8 \
    --clip_names \
        person20_handwaving_d2_uncomp.avi_189-203 \
    --pdf_zoom_region .33 .05 .53 .25 \
    --pdf_zoom_window_position .48 .48 .97 .97 \
    --exp_names "bi-TAI" \
    --model_labels "bi-TAI (ours)" \
    --dest_path "$TEMP_FIG_ROOT/KTH_T=5_bidirectional_pred_good" \
    --pdf_frame_width 1.65 \
    --pdf_frame_height 1.65 \
    --pdf_spacing 0.07

# kth_ablation_bidirectional_pred
python compare_intermediate_preds.py \
    --results_root results \
    --clip_names \
        person20_handwaving_d2_uncomp.avi_189-203 \
    --pdf_zoom_region .33 .05 .53 .25 \
    --ts 8 \
    --exp_names "bi-SA" "bi-TWA" "bi-TWI" "bi-TAI" \
    --model_labels "bi-SA" "bi-TWA" "bi-TWI" "bi-TAI" \
    --dataset_name KTH-test_data_list_T=5 \
    --dest_path "$TEMP_FIG_ROOT/KTH_T=5_bidirectional_pred_good" \
    bidirectional_pred

# kth_ablation_interp_net
python compare_intermediate_preds.py \
    --results_root results \
    --clip_names person20_handwaving_d2_uncomp.avi_189-203 \
    --pdf_zoom_region .33 .05 .53 .25 \
    --ts 8 \
    --exp_names "bi-TWI" "bi-TAI" \
    --model_labels "bi-TWI" "bi-TAI" \
    --dataset_name KTH-test_data_list_T=5 \
    --dest_path "$TEMP_FIG_ROOT/KTH_T=5_interp_net_good" \
    interp_net_pred

### RENAME/COPY FIGURES TO FINAL PAPER FIGURE DIRECTORY ###

mkdir -p paper_figs

cp $TEMP_FIG_ROOT/KTH_T=5_good/* paper_figs
cp $TEMP_FIG_ROOT/KTH_T=5_bad/* paper_figs
cp $TEMP_FIG_ROOT/UCF_T=3_good/* paper_figs
cp $TEMP_FIG_ROOT/UCF_T=3_bad/* paper_figs
cp $TEMP_FIG_ROOT/HMDB_T=3_good/* paper_figs
cp $TEMP_FIG_ROOT/HMDB_T=3_bad/* paper_figs
cp $TEMP_FIG_ROOT/Imagenet_T=3_good/* paper_figs

cp $TEMP_FIG_ROOT/KTH_T=5_bidirectional_pred_good/person20_handwaving_d2_uncomp.avi_189-203.pdf paper_figs/person20_handwaving_d2_uncomp.avi_189-203_gt_08.pdf
cp $TEMP_FIG_ROOT/KTH_T=5_bidirectional_pred_good/person20_handwaving_d2_uncomp.avi_189-203/08.pdf paper_figs/person20_handwaving_d2_uncomp.avi_189-203_bidirectional_pred_08.pdf
cp $TEMP_FIG_ROOT/KTH_T=5_interp_net_good/person20_handwaving_d2_uncomp.avi_189-203/08.pdf paper_figs/person20_handwaving_d2_uncomp.avi_189-203_interp_net_08.pdf

# Remove temporary directory
rm -r "$TEMP_FIG_ROOT"
