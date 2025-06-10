# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.distributed as dist

# --- NEW IMPORTS ---
import pandas as pd  # For reading sentence CSV
import torch.nn.functional as F  # For normalize
# --- END NEW IMPORTS ---

from base import BaseTrainer, Multi_BaseTrainer_dist
from model.model import sim_matrix
from utils import inf_loop
from pathlib import Path
import sys
import json

# --- ADDED GLOBAL CONFIG ---
# This needs to match the clip duration used when you split your video.
CLIP_DURATION_SECONDS = 5


# --- END ADDED GLOBAL CONFIG ---

class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )


class Multi_Trainer_dist_MIR(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    class Multi_Trainer_dist_MIR(Multi_BaseTrainer_dist):
        """
        Trainer class

        Note:
            Inherited from BaseTrainer.
        """

        def __init__(self, args, model, loss, metrics, optimizer, scheduler, gpu, config, data_loader,
                     valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                     visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
            super().__init__(args, model, loss, metrics, optimizer, scheduler, gpu, config, writer)
            self.config = config
            self.args = args
            self.data_loader = data_loader

            # --- MODIFIED SECTION START ---
            if len_epoch is None:
                # If the data_loader (training data) is empty, set len_epoch to 0.
                # This handles evaluation-only runs where no training data is provided.
                if not self.data_loader:
                    self.len_epoch = 0
                else:
                    # Original logic for epoch-based training: take the minimum length
                    # among multiple data loaders if they exist.
                    self.len_epoch = min([len(x) for x in data_loader])
            else:
                # Iteration-based training (original logic, not relevant for our current use case)
                self.data_loader = inf_loop(data_loader)
                self.len_epoch = len_epoch
            # --- MODIFIED SECTION END ---

            self.valid_data_loader = valid_data_loader
            self.do_validation = self.valid_data_loader is not None
            self.lr_scheduler = lr_scheduler
            self.visualizer = visualizer
            self.val_chunking = True

            # --- MODIFIED SECTION START (Defensive check for self.batch_size if data_loader is empty) ---
            if self.data_loader:  # Only access data_loader[0] if it's not empty
                self.batch_size = self.data_loader[0].batch_size
                self.log_step = int(np.sqrt(self.batch_size))
                self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
            else:  # For evaluation-only, set dummy values
                self.batch_size = 1  # Or any non-zero dummy value
                self.log_step = 1  # Or any non-zero dummy value
                self.total_batch_sum = 0  # No batches for training
            # --- MODIFIED SECTION END ---

            self.tokenizer = tokenizer
            self.max_samples_per_epoch = max_samples_per_epoch
            self.n_gpu = self.args.world_size
            self.allgather = AllGather_multi.apply
            # self.writer = writer

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            # if self.writer is not None:
            #     self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch, scaler, gpu):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        if dist.get_rank() == 0:
            Path(self.args.save_dir).mkdir(parents=True, exist_ok=True)
            stats_file = open(Path(self.args.save_dir) / str('stats_vtc.txt'), 'a', buffering=1)
            print(' '.join(sys.argv))
            print(' '.join(sys.argv), file=stats_file)

        log = {}

        if self.do_validation and epoch == 1:
            val_log = self._valid_epoch(0, gpu)
            if self.args.rank == 0:
                log.update(val_log)

        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))

        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding='max_length',
                                                  max_length=30,
                                                  truncation=True)
                data['text'] = {key: val.cuda(gpu, non_blocking=True) for key, val in data['text'].items()}
                data['video'] = data['video'].cuda(gpu, non_blocking=True)
                data['relation'] = data['relation'].cuda(gpu, non_blocking=True)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    with torch.cuda.amp.autocast():
                        loss, loss_dict, _ = self.model(data, self.allgather, self.n_gpu, self.args, self.config,
                                                        self.loss, gpu, task_names='Dual', dataset_name='epic')
                        assert loss == loss_dict['Dual']

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()

                if dist.get_rank() == 0:
                    if (batch_idx) % self.args.print_freq == 0:
                        stats = dict(epoch=epoch, step=batch_idx,
                                     lr_weights=self.optimizer.param_groups[0]['lr'],
                                     loss=loss.item())
                        print(json.dumps(stats), file=stats_file)

                if self.writer is not None and self.args.rank == 0:
                    # self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())
                    total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
                    current = batch_idx * self.data_loader[dl_idx].batch_size
                    final_total = (epoch - 1) * total + current
                    self.writer.add_scalar(f'Loss_training/loss_{dl_idx}', loss.detach().item(), final_total)

                total_loss[dl_idx] += loss.detach().item()

                # if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                if batch_idx % self.args.print_freq == 0 and self.args.rank == 0:
                    self.logger.info('Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))

                self.optimizer.zero_grad()

            # if batch_idx == 100:
            #    break

            if batch_idx == self.len_epoch:
                break

        log.update({f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))})

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Loss_training/loss_total_{dl_idx}', tl, epoch - 1)

        if self.do_validation:
            val_log = self._valid_epoch(epoch, gpu)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch(self, epoch, gpu):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        idx_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(tqdm(dl)):
                    meta_arr[dl_idx].append(data['meta'])

                    # Original logic for processing batch text and video
                    idx_embed = data['meta']['paths'].cuda(gpu, non_blocking=True)  # paths contain video_ids
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    data['text'] = {key: val.cuda(gpu, non_blocking=True) for key, val in data['text'].items()}
                    data['video'] = data['video'].cuda(gpu, non_blocking=True)

                    # Compute embeddings for the batch
                    ret = self.model.module.infer(data, return_embeds=True, task_names="Dual", ret={})
                    vid_embed = ret['video_embeds']
                    text_embed = ret['text_embeds']  # This text_embed is from the batch's queries

                    # --- Distributed data collection (necessary for `sim_matrix` over full dataset) ---
                    vid_embed_all = [torch.zeros_like(vid_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(vid_embed_all, vid_embed)
                    vid_embed_all = torch.cat(vid_embed_all, dim=0)

                    text_embed_all = [torch.zeros_like(text_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(text_embed_all, text_embed)
                    text_embed_all = torch.cat(text_embed_all, dim=0)

                    text_embed_arr[dl_idx].append(text_embed_all.cpu())
                    vid_embed_arr[dl_idx].append(vid_embed_all.cpu())

                    idx_embed_all = [torch.zeros_like(idx_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(idx_embed_all, idx_embed)
                    idx_embed_all = torch.cat(idx_embed_all, dim=0)
                    idx_embed_arr[dl_idx].append(idx_embed_all.cpu())
                    # --- End distributed data collection ---

            if self.writer is not None and self.args.rank == 0:
                for dl_idx in range(len(self.valid_data_loader)):
                    tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    self.writer.add_scalar(f'Loss_val/loss_total_{dl_idx}', tl, epoch - 1)

        for dl_idx in range(len(self.valid_data_loader)):
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            # Concatenate all gathered embeddings for the full dataset
            text_embeds_full = torch.cat(text_embed_arr[dl_idx])  # Shape: (total_queries, embed_dim)
            vid_embeds_full = torch.cat(vid_embed_arr[dl_idx])  # Shape: (total_videos, embed_dim)
            arr_embeds_full = torch.cat(idx_embed_arr[dl_idx])  # Shape: (total_videos, ) -> video_ids

            # Compute similarity matrix over the FULL validation dataset
            # Shape: (total_queries, total_videos)
            sims = sim_matrix(text_embeds_full, vid_embeds_full).detach().cpu().numpy()

            # --- METRIC CALCULATION (Keep as is, now calculates over your full mock dataset) ---
            for metric in self.metrics:
                metric_name = metric.__name__
                # `arr_embeds_full` (video_ids) is passed as `idx_arr` to `mir_metrics_vtc`
                res = metric(sims, arr_embeds_full)
                if self.args.rank == 0:
                    self.logger.info(verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                                             mode=metric_name, args=self.args))
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        key = key.replace('[', '_').replace(']', '_')
                        self.writer.add_scalar(f'Val_metrics_{dl_idx}/{key}', val, epoch - 1)

                if self.visualizer is not None and self.args.rank == 0:
                    # Note: visualizer.visualize_ranking might need access to raw text.
                    # It relies on `meta_arr_cat` which is accumulated.
                    # For this, `meta_arr` might need to accumulate `data['meta']['narration']` etc.
                    # But the default `meta_arr` should contain enough.
                    meta_arr_cat = {key: [] for key in meta_arr[0]}
                    for meta_batch in meta_arr:  # Iterate through each collected batch of meta
                        for key, val_list in meta_batch.items():
                            meta_arr_cat[key].extend(val_list)  # Extend lists for each key
                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)
            # --- END METRIC CALCULATION ---

            # --- CUSTOM PREDICTION PRINTING LOGIC START ---
            if self.args.rank == 0:  # Only print from the master process (GPU 0)
                print("\n--- Top Clip Retrieval Predictions for ALL Queries ---")

                # Load the full list of all queries from the sentence CSV (matching data loader order)
                sentence_csv_path = os.path.join(self.config['data_loader']['args']['meta_dir'],
                                                 "EPIC_100_retrieval_test_sentence.csv")
                df_sentence_csv = pd.read_csv(sentence_csv_path)
                # Create a dict from narration_id to actual query text for easy lookup
                all_narration_id_to_query_text = dict(
                    zip(df_sentence_csv['narration_id'], df_sentence_csv['narration']))

                # `arr_embeds_full` contains the list of all 5-second clip IDs in order.
                # It's a tensor, convert to list of strings
                all_5s_clip_ids_evaluated = [str(x) for x in
                                             arr_embeds_full.tolist()]  # Convert to string for consistent ID

                top_k_results_per_query = 5  # Number of top clips to display per query (adjust as desired)

                # Iterate through each query in the `sims` matrix (num_queries x num_clips)
                for query_row_idx in range(sims.shape[0]):  # Loop through each query
                    query_narration_id = df_sentence_csv['narration_id'].iloc[query_row_idx]
                    query_text = all_narration_id_to_query_text.get(query_narration_id, "UNKNOWN_QUERY")

                    print(f"\nQuery: '{query_text}' (ID: {query_narration_id})")
                    print("  Top 5s Clips:")

                    # Get similarity scores for this query across all 5s clips
                    scores_for_query = sims[query_row_idx, :]  # (1, num_5s_clips) row for this query

                    # Sort clips by score in descending order
                    ranked_clip_indices = np.argsort(scores_for_query)[
                                          ::-1]  # Get indices of clips from highest score to lowest

                    for rank, clip_col_idx in enumerate(ranked_clip_indices[:top_k_results_per_query]):
                        clip_id = all_5s_clip_ids_evaluated[clip_col_idx]
                        score = scores_for_query[clip_col_idx]

                        # Extract clip number and calculate approximate time (using the CLIP_DURATION_SECONDS)
                        try:
                            # clip_id is like 'clip_000', so split by '_' and take the last part
                            clip_num_str = clip_id.split('_')[-1]
                            clip_num = int(clip_num_str)
                            start_approx = clip_num * CLIP_DURATION_SECONDS
                            end_approx = (clip_num + 1) * CLIP_DURATION_SECONDS
                            time_str = f"({start_approx:.1f}s - {end_approx:.1f}s)"
                        except ValueError:
                            time_str = "(time N/A)"  # Fallback if clip_id format is unexpected

                        print(f"    Rank {rank + 1}: {clip_id} {time_str} | Score: {score:.4f}")

                print("\n--- End Top Clip Retrieval Predictions for ALL Queries ---")
            # --- END CUSTOM PREDICTION PRINTING LOGIC ---

        res_dict = {}
        if self.args.rank == 0:
            res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                        for dl_idx in range(len(self.valid_data_loader))}
            res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def verbose(epoch, metrics, mode, args, name="TEST"):
    if dist.get_rank() == 0:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        stats_file = open(Path(args.save_dir) / 'stats_vtc.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    ndcg_v2t, ndcg_tv2, ndcg_avg = metrics["nDCG_V2T"], metrics["nDCG_T2V"], metrics["nDCG_AVG"]
    map_v2t, map_t2v, map_avg = metrics["mAP_V2T"], metrics["mAP_T2V"], metrics["mAP_AVG"]

    msg = f"[{mode}]{name:s} epoch {epoch}, nDCG_V2T: {ndcg_v2t:.3f}, nDCG_T2V: {ndcg_tv2:.3f}, nDCG_AVG: {ndcg_avg:.3f},"
    msg += f", mAP_V2T: {map_v2t:.3f}, mAP_T2V: {map_t2v:.3f}, mAP_AVG: {map_avg:.3f}"

    print(msg)

    if dist.get_rank() == 0:
        stats = dict(epoch=epoch, msg=msg)
        print(json.dumps(stats), file=stats_file)

    return msg


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res