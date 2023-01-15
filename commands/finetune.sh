python main.py \
--features_path "bobsl/features/i3d_c2281_16f_m8_-15_4_d0.8_-3_22/" \
--gt_sub_path "bobsl/subtitles/manually-aligned/" \
--pr_sub_path "bobsl/subtitles/audio-aligned-heuristic-correction" \
--spottings_path "bobsl/spottings/annotations.pkl" \
--gpu_id 0 \
--batch_size 32 \
--n_workers 8 \
--pr_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--jitter_location \
--jitter_abs \
--jitter_loc_quantity 2. \
--load_words False \
--load_subtitles True \
--lr 1e-6 \
--save_path "inference_output/finetune_subtitles" \
--train_videos_txt "data/bobsl_align_train.txt" \
--val_videos_txt "data/bobsl_align_val.txt" \
--test_videos_txt "data/bobsl_align_test.txt" \
--n_epochs 100 \
--concatenate_prior True \
--min_sent_len_filter 0.5 \
--max_sent_len_filter 20 \
--shuffle_words_subs 0.5 \
--drop_words_subs 0.15 \
--resume "inference_output/train_coarse_subtitles/checkpoints/model_40_40.pt" \
--word-annotations M* D* \
--add-anchors-prior False \
--add-spottings-prior False \
--adjust-spottings-prior False \