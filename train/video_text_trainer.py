#-*- coding: utf-8 -*-

import time
from collections import defaultdict

import numpy as np
import os
import torch
from torch import nn
from tqdm import tqdm
from utils import F1Logger
from utils import seconds_to_string, F1Logger
from utils import colorize

from misc.evaluate_sub_alignment import eval_subtitle_alignment

import pickle 

from train.base_trainer import BaseTrainer

class VideoTextTrainer(BaseTrainer):

    def train(self, dataloader, mode='train', epoch=-1):

        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        self.dataloader = dataloader

        tb_stepmod = (100 if mode == 'train' else 1) if not self.opts.test_only else 1

        bs = self.opts.batch_size
        counter = 0

        # cummulative losses and metrics dictionary
        metrics_dict = defaultdict( float)  

        bar = tqdm(total=len(dataloader))

        self.data_tic = self.step_tic = None

        # Cumulative counters
        gt_vec_nonzeros = 0
        gt_vec_lengths = 0

        gt_vec_width = 0
        gt_vec_nb = 0

        if self.opts.concatenate_prior:

            pr_vec_nonzeros = 0
            pr_vec_lengths = 0

            pr_vec_empty = 0

            pr_vec_width = 0
            pr_vec_nb = 0

        if self.opts.add_anchors_prior:

            spottings_probs_nonzeros = 0
            spottings_probs_lengths = 0

            spottings_probs_empty = 0

        if self.opts.add_spottings_prior:

            spottings_prior_nonzeros = 0
            spottings_prior_lengths = 0

            spottings_prior_empty = 0

            spottings_width = 0
            spottings_nb = 0

        # -- for f1 accumulation
        self.f1_logger = F1Logger(overlaps=(0.1, 0.25, 0.5,))
        self.f1_logger_b = F1Logger(overlaps=(0.1, 0.25, 0.5,), suffix='_b')
        
        seen_out_files = []
        
        for b_id, batch_sample in enumerate(dataloader): 

            self.model.zero_grad()
            model_out = self.model.forward(batch_sample)
            # if b_id == 0: 
            #     preds_vec = model_out['preds'][0]
            #     prior_vec = model_out['pr_vec'][0]
            #     gt_vec = model_out['gt_vec'][0]
            #     # feats = model_out['feats'][0,15,:]
            #     print('average pred', preds_vec.round().mean())
            #     print('preds vs prior', (preds_vec.round()+prior_vec==2).sum()/((preds_vec.round()+prior_vec>=1).sum()+1e-5))
            #     print('preds vs gt', (preds_vec.round()+gt_vec==2).sum()/((preds_vec.round()+gt_vec>=1).sum()+1e-5))
            #     # print('feats', [f for ixf, f in enumerate(feats) if gt_vec[ixf]==1])
            #     # print('txt', model_out['txt'][0])

            ### in text mode, save subtitles and probabilties
            if mode == 'test':
                assert self.opts.batch_size == 1
                text = batch_sample['orig_txt'][0]
                if text == '': # don't allow empty text in vtt files
                    text = '.'
                video_fname = batch_sample['path'][0]

                ### if the video probs/vtt have not been saved already
                if video_fname not in seen_out_files:
                    if len(seen_out_files)>0 and self.opts.save_probs:
                        prev_video_fname = seen_out_files[-1]
                        ## save probabilities 
                        save_path = os.path.join(self.opts.save_probs_folder, prev_video_fname)
                        print('saving ', save_path)
                        os.makedirs(save_path, exist_ok=True)
                        pickle.dump(prob_dict, open(os.path.join(save_path, 'out.pkl'), 'wb'))

                    # make new object to save the predictions 
                    prob_dict = {'preds': [], 
                                'wind_fr_to': [], 
                                'txt': [], 
                                'probs': []}

                    seen_out_files.append(video_fname)

                    if self.opts.save_vtt: 
                        out_folder = os.path.join(self.opts.save_subs_folder, video_fname)
                        out_file = os.path.join(self.opts.save_subs_folder, video_fname, 'signhd.vtt')
                        print('saving ', out_file)
                        os.makedirs(out_folder, exist_ok=True)
                        fw = open(out_file, 'w', encoding="utf-8")
                        fw.write('WEBVTT\n\n')

                # stuff to save from dataloader 
                wind_fr_to = batch_sample['wind_fr_to'][0].numpy()
                prob_dict['wind_fr_to'].append(wind_fr_to)
                prob_dict['txt'].append(text)

                # stuff to save from model output 
                preds_vec = model_out['preds'][0]
                prob_dict['preds'].append(preds_vec)

                if self.opts.save_vtt: 
                    time_len_window = wind_fr_to[1] - wind_fr_to[0]
                    frame_len_window = len(preds_vec)
                    preds_vec_round = np.round(preds_vec)

                    if np.max(preds_vec_round) < 1: # no subtitle predictions 
                        pred_fr_sec = batch_sample['pr_fr_to'][0].numpy()[0]
                        pred_to_sec = batch_sample['pr_fr_to'][0].numpy()[1]

                    else:
                        pred_fr_sec = wind_fr_to[0]+np.where(preds_vec_round==1)[0][0]/frame_len_window*time_len_window
                        pred_to_sec = wind_fr_to[0]+np.where(preds_vec_round==1)[0][-1]/frame_len_window*time_len_window
                    

                    pred_fr_sec = float(pred_fr_sec)
                    pred_to_sec = float(pred_to_sec)

                    pred_str_fr = seconds_to_string(pred_fr_sec)
                    pred_str_to = seconds_to_string(pred_to_sec)
                    pred_str_fr_to = f'{pred_str_fr} --> {pred_str_to}'

                    fw.write(f'{pred_str_fr_to}\n')
                    fw.write(f'{text}\n\n')

            # ------------------------- Time steps  -------------------------

            if self.step_tic:
                metrics_dict['t'] += time.time() - self.step_tic
                if self.data_tic is not None:
                    metrics_dict['dt'] += time.time() - self.data_tic
                    metrics_dict['dt/total'] = (
                        metrics_dict['dt'] / metrics_dict['t']) * (counter + 1)
            self.step_tic = time.time()

            # ------------------------- Loss  -------------------------
            loss = model_out['loss'].mean()

            # ------------------------- Backprop  -------------------------
            if mode == 'train':
                loss.backward(retain_graph=False)

                if self.opts.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                                self.opts.grad_clip_norm)

                self.optimizer.step()

            # ------------------------- Metrics  -------------------------

            metrics_dict['loss'] += model_out['loss'].mean().detach().cpu().item()
            if 'iou' in model_out:
                metrics_dict['iou'] += model_out['iou']
            if 'base_iou' in model_out:
                metrics_dict['base_iou'] += model_out['base_iou']

            # metrics for prediction
            if 'tp' in model_out:
                self.f1_logger.update(model_out)
                metrics_dict['frame_acc'] = self.f1_logger.accuracy * (counter + 1) # hack 
                f1s, overlaps = self.f1_logger.f1
                for f1, ov in zip(f1s, overlaps):
                    metrics_dict[f'f1@{ov}'] = f1 * (counter+1) # hack

            # metrics for base
            if 'tp_b' in model_out:
                self.f1_logger_b.update(model_out)
                metrics_dict['frame_acc_b'] = self.f1_logger_b.accuracy * (counter + 1) # hack 
                f1s, overlaps = self.f1_logger_b.f1
                for f1, ov in zip(f1s, overlaps):
                    metrics_dict[f'f1@{ov}_b'] = f1 * (counter+1) # hack

            # - tb summaries
            if (self.opts.test_only or
                    mode == 'train' and ( b_id % tb_stepmod) == 0
                        or b_id == len(dataloader) - 1):
                for loss_name, loss_val in metrics_dict.items():
                    self.tb_writer.add_scalar(f'{mode}/{loss_name}',
                                            loss_val / (counter + 1),
                                            self.global_step)

            counter += 1
            if mode == 'train' or self.opts.test_only:
                self.global_step += 1

            bar.update(1)

            desc = "%s: " % mode
            for cuml_name, cuml in sorted(metrics_dict.items()):
                desc += "%s %.2f " % (cuml_name, cuml / counter)
            bar.set_description(desc)

            self.data_tic = time.time(
            )  # this counts how long we are waiting for data

            # Update cumulative counters
            gt_vec_nonzeros += model_out["gt_vec_nonzero"]
            gt_vec_lengths += model_out["gt_vec_length"]

            gt_vec_width += model_out["gt_vec_width"]
            gt_vec_nb += model_out["gt_vec_nb"]

            if self.opts.concatenate_prior:

                pr_vec_nonzeros += model_out["pr_vec_nonzero"]
                pr_vec_lengths += model_out["pr_vec_length"]

                pr_vec_empty += model_out["pr_vec_empty"]

                pr_vec_width += model_out["pr_vec_width"]
                pr_vec_nb += model_out["pr_vec_nb"]

            if self.opts.add_anchors_prior:

                spottings_probs_nonzeros += model_out["spottings_probs_nonzero"]
                spottings_probs_lengths += model_out["spottings_probs_length"]

                spottings_probs_empty += model_out["spottings_probs_empty"]

            if self.opts.add_spottings_prior:

                spottings_prior_nonzeros += model_out["spottings_prior_nonzero"]
                spottings_prior_lengths += model_out["spottings_prior_length"]

                spottings_prior_empty += model_out["spottings_prior_empty"]

                spottings_width += model_out["spottings_width"]
                spottings_nb += model_out["spottings_nb"]

        bar.close()

        if mode=='test' and self.opts.save_probs:
            ### save last video at epoch end 
            prev_video_fname = seen_out_files[-1]
            ## save probabilities 
            save_path = os.path.join(self.opts.save_probs_folder, prev_video_fname)
            print('saving ', save_path)
            os.makedirs(save_path, exist_ok=True)
            pickle.dump(prob_dict, open(os.path.join(save_path, 'out.pkl'), 'wb'))
    
        desc = "Epoch end: %s: " % mode
        for cuml_name, cuml in sorted(metrics_dict.items()):
            desc += "%s %.2f " % (cuml_name, cuml / counter)
            self.tb_writer.add_scalar(f'{mode}_epoch/{cuml_name}',
                                      cuml / counter, self.global_step)

        # Print sanity checks
        gt_vec_zero_prop = np.around(100 * (gt_vec_nonzeros / gt_vec_lengths), 2)        
        desc += f" gt prop of nonzero frames: {gt_vec_zero_prop}%"

        gt_vec_width_avg = np.around(gt_vec_width / gt_vec_nb, 2)
        desc += f" avg width of gt vec: {gt_vec_width_avg}"

        if self.opts.concatenate_prior:

            pr_vec_zero_prop = np.around(
                100 * (pr_vec_nonzeros / pr_vec_lengths),
                2
            )
            desc += f" audio pr prop of nonzero frames: {pr_vec_zero_prop}%"

            # Should be zero (we always have some audio prior in the windows)
            # pr_vec_empty_prop = np.around(
            #     100 * (pr_vec_empty / len(dataloader.dataset)),
            #     2
            # )
            # desc += f" audio pr prop of zero vectors: {pr_vec_empty_prop}%"

            pr_vec_width_avg = np.around(pr_vec_width / pr_vec_nb, 2)
            desc += f" avg width of pr vec: {pr_vec_width_avg}"

        if self.opts.add_anchors_prior:

            spottings_probs_zero_prop = np.around(
                100 * (spottings_probs_nonzeros / spottings_probs_lengths),
                2
            )
            desc += f" spottings probs prop of nonzero frames: {spottings_probs_zero_prop}%"

            spottings_probs_empty_prop = np.around(
                100 * (spottings_probs_empty / len(dataloader.dataset)),
                2
            )
            desc += f" spottings prop of zero vectors: {spottings_probs_empty_prop}%"

        if self.opts.add_spottings_prior:

            spottings_prior_zero_prop = np.around(
                100 * (spottings_prior_nonzeros / spottings_prior_lengths),
                2
            )
            desc += f" spottings prior prop of nonzero frames: {spottings_prior_zero_prop}%"

            if not self.opts.add_anchors_prior:  # same value
                spottings_prior_empty_prop = np.around(
                    100 * (spottings_prior_empty / len(dataloader.dataset)),
                    2
                )
                desc += f" spottings prop of zero vectors: {spottings_prior_empty_prop}%"

            spottings_width_avg = np.around(spottings_width / spottings_nb, 2)
            desc += f" avg width of spottings prior: {spottings_width_avg}"

        print(desc)
        self.tb_writer.flush()

        if counter > 0:
            val_metric = metrics_dict['frame_acc'] / counter
        else:
            val_metric = 0
        return bar.desc, val_metric


