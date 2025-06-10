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
import pandas as pd
import torch.nn.functional as F
# --- END NEW IMPORTS ---

# --- MODIFIED IMPORT FOR BASE CLASS ---
import base
# --- END MODIFIED IMPORT ---

from model.model import sim_matrix
from utils import inf_loop
from pathlib import Path
import sys
import json

# --- ADDED GLOBAL CONFIG ---
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


class Multi_Trainer_dist_MIR(base.Multi_BaseTrainer_dist):
    """
    Trainer class
    """

    def __init__(self, args, model, loss, metrics, optimizer, scheduler, gpu, config,
                 data_loader=None,
                 valid_data_loader=None,
                 lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, scheduler, gpu, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader

        if len_epoch is None:
            if not self.data_loader:
                self.len_epoch = 0
            else:
                self.len_epoch = min([len(x) for x in data_loader])
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True

        if self.data_loader:
            self.batch_size = self.data_loader[0].batch_size
            self.log_step = int(np.sqrt(self.batch_size))
            self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        else:
            self.batch_size = 1
            self.log_step = 1
            self.total_batch_sum = 0

        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply

        # --- NEW: Initialize accumulators as class attributes ---
        # These will be lists of lists, holding collected data per DataLoader index.
        self.text_embed_arr_storage = {x: [] for x in range(len(self.valid_data_loader))}
        self.vid_embed_arr_storage = {x: [] for x in range(len(self.valid_data_loader))}
        self.idx_embed_arr_storage = {x: [] for x in range(len(self.valid_data_loader))}
        self.meta_arr_storage = {x: {} for x in range(len(self.valid_data_loader))}  # For meta data
        # --- END NEW ---

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch, scaler, gpu):
        # This function is not primarily used for zero-shot evaluation (epochs=0)

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
            if hasattr(loader, 'train_sampler'):
                loader.train_sampler.set_epoch(epoch)

        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            # This entire loop will be skipped if self.data_loader is empty
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                # ... (original training logic) ...
                pass

            if batch_idx == self.len_epoch:
                break

        # Log updates related to total_loss (will be 0 if no training batches)
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
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)

        # --- NEW: Clear accumulators at the start of each validation epoch ---
        for dl_idx in range(len(self.valid_data_loader)):
            self.text_embed_arr_storage[dl_idx].clear()
            self.vid_embed_arr_storage[dl_idx].clear()
            self.idx_embed_arr_storage[dl_idx].clear()
            # Clear metadata storage
            for key in self.meta_arr_storage[dl_idx].keys():
                self.meta_arr_storage[dl_idx][key].clear()
        # --- END NEW ---

        with torch.no_grad():
            for dl_idx, dl in enumerate(self.valid_data_loader):
                # Ensure meta_arr_storage for this dl_idx has lists for expected keys if not already (from init)
                # This ensures keys are present even if a batch is empty or only some keys appear in first batch
                # The keys available in 'meta' usually depend on the DataLoader's implementation (e.g. EPIC-Kitchens_MIR_dataset.py)
                # Common keys are 'narration_id', 'narration', 'video_id', 'paths', 'start_frame', 'stop_frame'
                expected_meta_keys = ['narration_id', 'participant_id', 'video_id', 'narration', 'paths',
                                      'raw_captions']
                for key in expected_meta_keys:
                    if key not in self.meta_arr_storage[dl_idx]:  # Initialize if not present
                        self.meta_arr_storage[dl_idx][key] = []

                for batch_idx, data in enumerate(tqdm(dl)):
                    # Accumulate metadata for each batch
                    # `data['meta']` is a dictionary of lists for the current batch
                    for k, v_list in data['meta'].items():
                        if k in self.meta_arr_storage[dl_idx]:
                            self.meta_arr_storage[dl_idx][k].extend(v_list)
                        else:  # Handle new keys if they appear (unlikely but safe)
                            self.meta_arr_storage[dl_idx][k] = v_list

                    # Original logic for processing batch text and video
                    # data['meta']['paths'] is now a tensor of numerical IDs (from dataset)
                    idx_embed = data['meta']['paths'].cuda(gpu, non_blocking=True)
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    data['text'] = {key: val.cuda(gpu, non_blocking=True) for key, val in data['text'].items()}
                    data['video'] = data['video'].cuda(gpu, non_blocking=True)

                    ret = self.model.module.infer(data, return_embeds=True, task_names="Dual", ret={})
                    vid_embed = ret['video_embeds']
                    text_embed = ret['text_embeds']

                    # Distributed data collection (necessary for `sim_matrix` over full dataset)
                    vid_embed_all = [torch.zeros_like(vid_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(vid_embed_all, vid_embed)
                    vid_embed_all = torch.cat(vid_embed_all, dim=0)

                    text_embed_all = [torch.empty_like(text_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(text_embed_all, text_embed)
                    text_embed_all = torch.cat(text_embed_all, dim=0)

                    idx_embed_all = [torch.empty_like(idx_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(idx_embed_all, idx_embed)
                    idx_embed_all = torch.cat(idx_embed_all, dim=0)

                    # Append collected embeddings to class attributes
                    self.text_embed_arr_storage[dl_idx].append(text_embed_all.cpu())
                    self.vid_embed_arr_storage[dl_idx].append(vid_embed_all.cpu())
                    self.idx_embed_arr_storage[dl_idx].append(idx_embed_all.cpu())
                    # --- End distributed data collection ---

            if self.writer is not None and self.args.rank == 0:
                for dl_idx in range(len(self.valid_data_loader)):
                    tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    self.writer.add_scalar(f'Loss_val/loss_total_{dl_idx}', tl, epoch - 1)

        for dl_idx in range(len(self.valid_data_loader)):
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            # Concatenate all gathered embeddings from storage for the full dataset
            text_embeds_full = torch.cat(self.text_embed_arr_storage[dl_idx])
            vid_embeds_full = torch.cat(self.vid_embed_arr_storage[dl_idx])
            arr_embeds_full = torch.cat(self.idx_embed_arr_storage[dl_idx])

            # Compute similarity matrix over the FULL validation dataset
            sims = sim_matrix(text_embeds_full, vid_embeds_full).detach().cpu().numpy()

            # --- METRIC CALCULATION (Keep as is, now calculates over your full mock dataset) ---
            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(sims, arr_embeds_full)  # arr_embeds_full contains numerical video_ids now
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
                    # meta_arr_cat needs to be a dictionary of lists.
                    # It's accumulated from self.meta_arr_storage.
                    meta_arr_cat_flattened = {}
                    # `self.meta_arr_storage[dl_idx]` is a dictionary of lists (e.g., {'narration_id': [...], 'video_id': [...]})
                    for key, values_list in self.meta_arr_storage[dl_idx].items():
                        meta_arr_cat_flattened[key] = values_list

                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat_flattened, nested_metrics)
            # --- END METRIC CALCULATION ---

            # --- CUSTOM PREDICTION PRINTING LOGIC START ---
            if self.args.rank == 0:
                print("\n--- Top Clip Retrieval Predictions for ALL Queries ---")

                sentence_csv_path = os.path.join(self.config['data_loader']['args']['meta_dir'],
                                                 "EPIC_100_retrieval_test_sentence.csv")
                df_sentence_csv = pd.read_csv(sentence_csv_path)
                all_narration_id_to_query_text = dict(
                    zip(df_sentence_csv['narration_id'], df_sentence_csv['narration']))

                # Retrieve the numerical_idx_to_video_id mapping from the dataset object
                # This needs to be accessible. The easiest way is to pass it from dataset to trainer init
                # or retrieve it from the dataset instance in valid_data_loader[dl_idx].dataset
                dataset_instance = self.valid_data_loader[dl_idx].dataset
                if hasattr(dataset_instance, 'numerical_idx_to_video_id'):
                    numerical_idx_to_video_id_map = dataset_instance.numerical_idx_to_video_id
                else:
                    print("Warning: numerical_idx_to_video_id map not found in dataset. Cannot print clip IDs by name.")
                    numerical_idx_to_video_id_map = {}  # Fallback to empty map

                # `arr_embeds_full` contains the numerical video IDs. Convert them to string IDs for printing.
                # Use a map function for robustness.
                all_5s_clip_ids_evaluated_numerical = arr_embeds_full.tolist()  # These are the numerical IDs
                all_5s_clip_ids_evaluated_strings = [
                    numerical_idx_to_video_id_map.get(idx, f"UNKNOWN_VID_{idx}")
                    for idx in all_5s_clip_ids_evaluated_numerical
                ]

                top_k_results_per_query = 5

                for query_row_idx in range(sims.shape[0]):
                    query_narration_id = df_sentence_csv['narration_id'].iloc[query_row_idx]
                    query_text = all_narration_id_to_query_text.get(query_narration_id, "UNKNOWN_QUERY")

                    print(f"\nQuery: '{query_text}' (ID: {query_narration_id})")
                    print("  Top 5s Clips:")

                    scores_for_query = sims[query_row_idx, :]

                    ranked_clip_indices = np.argsort(scores_for_query)[::-1]

                    for rank, clip_col_idx in enumerate(ranked_clip_indices[:top_k_results_per_query]):
                        clip_id_numerical = all_5s_clip_ids_evaluated_numerical[clip_col_idx]
                        clip_id_string = numerical_idx_to_video_id_map.get(clip_id_numerical,
                                                                           f"UNKNOWN_VID_{clip_id_numerical}")
                        score = scores_for_query[clip_col_idx]

                        try:
                            # clip_id_string is like 'aria_P01_000'. Split by '_' and take the last numerical part.
                            clip_num_str = clip_id_string.split('_')[-1]
                            clip_num = int(clip_num_str)
                            start_approx = clip_num * CLIP_DURATION_SECONDS
                            end_approx = (clip_num + 1) * CLIP_DURATION_SECONDS
                            time_str = f"({start_approx:.1f}s - {end_approx:.1f}s)"
                        except ValueError:
                            time_str = "(time N/A)"

                        print(f"    Rank {rank + 1}: {clip_id_string} {time_str} | Score: {score:.4f}")

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
        if len(self.data_loader) > dl_idx and hasattr(self.data_loader[dl_idx], 'n_samples'):
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