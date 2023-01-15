python main.py \
--features_path "bobsl/features/i3d_c2281_16f_m8_-15_4_d0.8_-3_22/" \
--gt_sub_path "bobsl/subtitles/manually-aligned/" \
--pr_sub_path "bobsl/subtitles/audio-aligned-heuristic-correction" \
--spottings_path "bobsl/spottings/annotations.pkl" \
--gpu_id 0 \
--n_workers 8 \
--batch_size 1 \
--pr_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--centre_window \
--test_only \
--save_vtt True \
--save_probs True \
--dtw_postpro True \
--resume "bobsl/checkpoints/finetune_subtitles/checkpoints/model_40_40_50.pt" \
--word-annotations M* D* \
--add-anchors-prior False \
--add-spottings-prior False \
--adjust-spottings-prior False \

# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 70.89 F1@0.10: 74.08 F1@0.25: 66.78 F1@0.50: 53.22

### 2.7s shift baseline
# frame_accuracy: 62.40
# f1_10: 72.77
# f1_25: 64.08
# f1_50: 44.60
