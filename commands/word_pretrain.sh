python main.py \
--features_path "bobsl/features/i3d_c2281_16f_m8_-15_4_d0.8_-3_22/" \
--spottings_path "bobsl/spottings/annotations.pkl" \
--gpu_id 0 \
--batch_size 32 \
--n_workers 8 \
--pr_subs_delta_bias 0 \
--fixed_feat_len 20 \
--jitter_location \
--jitter_abs \
--jitter_loc_quantity 10. \
--load_words True \
--load_subtitles False \
--lr 1e-5 \
--centre_window \
--save_path "inference_output/word_pretrain" \
--train_videos_txt "data/bobsl_train_1658.txt" \
--val_videos_txt "data/bobsl_val_32.txt" \
--test_videos_txt "data/bobsl_test_250.txt" \
--pos_weight 19. \
--n_epochs 41 \
--save_every_n 5 \
--shuffle_getitem True \
--concatenate_prior True \
--random_subset_data 60 \
--word-annotations M* D* \
--add-anchors-prior False \
--add-spottings-prior False \
--adjust-spottings-prior False \