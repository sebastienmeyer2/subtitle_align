python main.py \
--gpu_id 0 \
--batch_size 64 \
--n_workers 32 \
--pr_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--jitter_location \
--jitter_abs \
--jitter_loc_quantity 2. \
--load_words False \
--load_subtitles True \
--lr 1e-6 \
--save_path 'inference_output/finetune_subtitles' \
--train_videos_txt 'data/bobsl_align_train.txt' \
--val_videos_txt 'data/bobsl_align_test.txt' \
--test_videos_txt 'data/bobsl_test_254.txt' \
--n_epochs 100 \
--concatenate_prior True \
--min_sent_len_filter 0.5 \
--max_sent_len_filter 20 \
--shuffle_words_subs 0.5 \
--drop_words_subs 0.15 \
--resume 'inference_output/train_coarse_subtitles/checkpoints/model_0000250341.pt' \