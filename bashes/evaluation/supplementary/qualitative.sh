#!/bin/bash

# KTH
python generate_comparison_videos.py \
    --results_root="results/KTH-test_data_list_T=5" \
    --exp_names MCnet Newson SuperSloMo bi-TAI \
    --exp_labels  MCnet Newson "Super SloMo" bi-TAI \
    --save_root="supplementary/KTH-test_data_list_T=5" \
    --clip_names \
        person17_handwaving_d1_uncomp.avi_397-411 \
        person18_handwaving_d4_uncomp.avi_463-477 \
        person19_handwaving_d1_uncomp.avi_341-355 \
        person19_handwaving_d4_uncomp.avi_296-310 \
        person19_jogging_d1_uncomp.avi_16-30 \
        person20_handwaving_d2_uncomp.avi_179-193 \
        person25_running_d4_uncomp.avi_263-277

python generate_comparison_videos.py \
    --results_root="results/KTH-test_data_list_T=10" \
    --exp_names MCnet Newson SuperSloMo bi-TAI \
    --exp_labels  MCnet Newson "Super SloMo" bi-TAI \
    --save_root="supplementary/KTH-test_data_list_T=10" \
    --clip_names \
        person17_handclapping_d4_uncomp.avi_103-122 \
        person17_jogging_d4_uncomp.avi_31-50 \
        person18_handwaving_d4_uncomp.avi_399-418 \
        person19_running_d2_uncomp.avi_278-297 \
        person20_handwaving_d1_uncomp.avi_156-175 \
        person20_walking_d3_uncomp.avi_442-461 \
        person21_handwaving_d4_uncomp.avi_535-554

# UCF-101
python generate_comparison_videos.py \
    --results_root="results/UCF-test_data_list_T=3" \
    --exp_names MCnet Newson SuperSloMo_val_test bi-TAI \
    --exp_labels  MCnet Newson "Super SloMo" bi-TAI \
    --save_root="supplementary/UCF-test_data_list_T=3" \
    --clip_names \
        v_CuttingInKitchen_g01_c02.avi_1-11 \
        v_FrisbeeCatch_g02_c03.avi_1-11 \
        v_LongJump_g06_c04.avi_1-11 \
        v_MoppingFloor_g03_c01.avi_1-11 \
        v_PoleVault_g03_c03.avi_1-11 \
        v_TennisSwing_g01_c01.avi_1-11 \
        v_WritingOnBoard_g04_c02.avi_1-11

python generate_comparison_videos.py \
    --results_root="results/UCF-test_data_list_T=5" \
    --exp_names MCnet Newson SuperSloMo_val_test bi-TAI \
    --exp_labels  MCnet Newson "Super SloMo" bi-TAI \
    --save_root="supplementary/UCF-test_data_list_T=5" \
    --clip_names \
        v_CuttingInKitchen_g01_c02.avi_1-13 \
        v_FrisbeeCatch_g02_c03.avi_1-13 \
        v_LongJump_g06_c04.avi_1-13 \
        v_MoppingFloor_g03_c01.avi_1-13 \
        v_PoleVault_g03_c03.avi_1-13 \
        v_TennisSwing_g01_c01.avi_1-13 \
        v_WritingOnBoard_g04_c02.avi_1-13

# HMDB-51
python generate_comparison_videos.py \
    --results_root="results/HMDB-test_data_list_T=3" \
    --exp_names MCnet Newson SuperSloMo_val_test bi-TAI \
    --exp_labels  MCnet Newson "Super SloMo" bi-TAI \
    --save_root="supplementary/HMDB-test_data_list_T=3" \
    --clip_names \
        5_Min_Tone_Abs_Workout_2__Fitness_Training_w__Tammy_situp_f_nm_np1_fr_goo_3.avi_1-11 \
        5_Min_Tone_Abs_Workout_2__Fitness_Training_w__Tammy_situp_f_nm_np1_le_goo_5.avi_1-11 \
        Bottoms_Up_-_Bartending_Lesson__Licor_43_Dreamsicle_pour_u_nm_np2_fr_goo_0.avi_1-11 \
        handstands_1_handstand_f_cm_np1_le_med_3.avi_1-11 \
        Muso_Jikiden_Eishinryu_in_Guldental_draw_sword_f_cm_np1_ba_med_2.avi_1-11 \
        ReggieMillerTakesonThreeAverageGuysinaShootout_shoot_ball_u_nm_np1_ba_med_3.avi_1-11 \
        St__Louis_Goalkeeping__Academy_elite_training_jump_f_nm_np1_ri_bad_10.avi_1-11

python generate_comparison_videos.py \
    --results_root="results/HMDB-test_data_list_T=5" \
    --exp_names MCnet Newson SuperSloMo_val_test bi-TAI \
    --exp_labels  MCnet Newson "Super SloMo" bi-TAI \
    --save_root="supplementary/HMDB-test_data_list_T=5" \
    --clip_names \
        5_Min_Tone_Abs_Workout_2__Fitness_Training_w__Tammy_situp_f_nm_np1_fr_goo_3.avi_1-13 \
        5_Min_Tone_Abs_Workout_2__Fitness_Training_w__Tammy_situp_f_nm_np1_le_goo_5.avi_1-13 \
        Bottoms_Up_-_Bartending_Lesson__Licor_43_Dreamsicle_pour_u_nm_np2_fr_goo_0.avi_1-13 \
        handstands_1_handstand_f_cm_np1_le_med_3.avi_1-13 \
        Muso_Jikiden_Eishinryu_in_Guldental_draw_sword_f_cm_np1_ba_med_2.avi_1-13 \
        ReggieMillerTakesonThreeAverageGuysinaShootout_shoot_ball_u_nm_np1_ba_med_3.avi_1-13 \
        St__Louis_Goalkeeping__Academy_elite_training_jump_f_nm_np1_ri_bad_10.avi_1-13

# ImageNet-VID
python generate_comparison_videos.py \
    --results_root="results/Imagenet-test_data_list_T=3" \
    --exp_names MCnet Newson SuperSloMo_val_test bi-TAI \
    --exp_labels  MCnet Newson "Super SloMo" bi-TAI \
    --save_root="supplementary/Imagenet-test_data_list_T=3" \
    --clip_names \
        ILSVRC2015_test_00027005.mkv_1-11 \
        ILSVRC2015_test_00076026.mkv_1-11 \
        ILSVRC2015_test_00147000.mkv_1-11 \
        ILSVRC2015_test_00166000.mkv_1-11 \
        ILSVRC2015_test_00171000.mkv_1-11 \
        ILSVRC2015_test_00199002.mkv_1-11

python generate_comparison_videos.py \
    --results_root="results/Imagenet-test_data_list_T=5" \
    --exp_names MCnet Newson SuperSloMo_val_test bi-TAI \
    --exp_labels  MCnet Newson "Super SloMo" bi-TAI \
    --save_root="supplementary/Imagenet-test_data_list_T=5" \
    --clip_names \
        ILSVRC2015_test_00027005.mkv_1-13 \
        ILSVRC2015_test_00076026.mkv_1-13 \
        ILSVRC2015_test_00147000.mkv_1-13 \
        ILSVRC2015_test_00166000.mkv_1-13 \
        ILSVRC2015_test_00171000.mkv_1-13 \
        ILSVRC2015_test_00199002.mkv_1-13