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
import pandas as pd  # Added for reading sentence CSV
import torch.nn.functional as F  # Added for normalize and other torch functions
# --- END NEW IMPORTS ---

# --- MODIFIED IMPORT FOR BASE CLASS ---
import base  # Corrected: import base module for base.Multi_BaseTrainer_dist
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


# --- MODIFIED CLASS INHERITANCE ---
class Multi_Trainer_dist_MIR(base.Multi_BaseTrainer_dist):  # Corrected: Inherit from base.Multi_BaseTrainer_dist
    # --- END MODIFIED CLASS INHERITANCE ---
    """
    Trainer class
    """

    def __init__(self, args, model, loss, metrics, optimizer, scheduler, gpu, config,
                 data_loader=None,  # Corrected: Made keyword arg with default
                 valid_data_loader=None,  # Corrected: Made keyword arg with default
                 lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, scheduler, gpu, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader

        if len_epoch is None:
            if not self.data_loader:  # Corrected: Handle empty data_loader
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

        if self.data_loader:  # Corrected: Handle empty data_loader
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

        # --- CORRECTED & NEW: Initialize accumulators as class attributes (self. prefix) ---
        self.text_embed_arr_storage = {x: [] for x in range(len(self.valid_data_loader))}
        self.vid_embed_arr_storage = {x: [] for x in range(len(self.valid_data_loader))}
        self.idx_embed_arr_storage = {x: [] for x in range(len(self.valid_data_loader))}
        self.meta_arr_storage = {x: {} for x in range(len(self.valid_data_loader))}
        # --- END CORRECTED & NEW ---

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

        # --- CORRECTED: Clear accumulators for the current validation epoch ---
        for dl_idx in range(len(self.valid_data_loader)):
            self.text_embed_arr_storage[dl_idx].clear()
            self.vid_embed_arr_storage[dl_idx].clear()
            self.idx_embed_arr_storage[dl_idx].clear()
            # Clear metadata storage by re-initializing lists for keys
            expected_meta_keys = ['narration_id', 'participant_id', 'video_id', 'narration', 'paths', 'raw_captions']
            self.meta_arr_storage[dl_idx] = {key: [] for key in expected_meta_keys}
        # --- END CORRECTED ---

        with torch.no_grad():
            for dl_idx, dl in enumerate(tqdm(self.valid_data_loader)):
                for batch_idx, data in enumerate(tqdm(dl, leave=False)):
                    # Accumulate metadata for each batch
                    for k, v_list in data['meta'].items():
                        if k in self.meta_arr_storage[dl_idx]:
                            self.meta_arr_storage[dl_idx][k].extend(v_list)
                        else:
                            self.meta_arr_storage[dl_idx][k] = v_list

                    idx_embed_local = data['meta']['paths'].cuda(gpu, non_blocking=True)
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    data['text'] = {key: val.cuda(gpu, non_blocking=True) for key, val in data['text'].items()}
                    data['video'] = data['video'].cuda(gpu, non_blocking=True)

                    ret = self.model.module.infer(data, return_embeds=True, task_names="Dual", ret={})
                    vid_embed_local = ret['video_embeds']
                    text_embed_local = ret['text_embeds']

                    # Append local embeddings directly to storage (on CPU)
                    self.text_embed_arr_storage[dl_idx].append(text_embed_local.cpu())
                    self.vid_embed_arr_storage[dl_idx].append(vid_embed_local.cpu())
                    self.idx_embed_arr_storage[dl_idx].append(idx_embed_local.cpu())

            if self.writer is not None and self.args.rank == 0:
                for dl_idx in range(len(self.valid_data_loader)):
                    tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    self.writer.add_scalar(f'Loss_val/loss_total_{dl_idx}', tl, epoch - 1)

        for dl_idx in range(len(self.valid_data_loader)):
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            # Concatenate all locally collected embeddings for this dataloader index
            text_embeds_local_concat = torch.cat(self.text_embed_arr_storage[dl_idx])
            vid_embeds_local_concat = torch.cat(self.vid_embed_arr_storage[dl_idx])
            idx_embeds_local_concat = torch.cat(self.idx_embed_arr_storage[dl_idx])

            # Perform final all_gather and de-duplication on Rank 0
            if self.n_gpu > 1:
                text_embeds_all_gpus = [torch.empty_like(text_embeds_local_concat) for _ in range(self.n_gpu)]
                torch.distributed.all_gather(text_embeds_all_gpus, text_embeds_local_concat)
                text_embeds_full = torch.cat(text_embeds_all_gpus, dim=0)

                vid_embeds_all_gpus = [torch.empty_like(vid_embeds_local_concat) for _ in range(self.n_gpu)]
                torch.distributed.all_gather(vid_embeds_all_gpus, vid_embeds_local_concat)
                vid_embeds_full = torch.cat(vid_embeds_all_gpus, dim=0)

                idx_embeds_all_gpus = [torch.empty_like(idx_embeds_local_concat) for _ in range(self.n_gpu)]
                torch.distributed.all_gather(idx_embeds_all_gpus, idx_embeds_local_concat)
                arr_embeds_full = torch.cat(idx_embeds_all_gpus, dim=0)

                # De-duplication of gathered tensors (assuming perfect order for slicing)
                sentence_csv_path_for_slice = os.path.join(self.config['data_loader']['args']['meta_dir'],
                                                           "EPIC_100_retrieval_test_sentence.csv")
                df_sentence_csv_for_slice = pd.read_csv(sentence_csv_path_for_slice)
                total_unique_queries_dataset = df_sentence_csv_for_slice.shape[0]

                # Get total unique video clips from dataset object
                total_unique_videos_dataset = len(self.valid_data_loader[dl_idx].dataset.video_id_to_numerical_idx)

                if self.args.rank == 0:
                    text_embeds_full_unique = text_embeds_full[:total_unique_queries_dataset]
                    vid_embeds_full_unique = vid_embeds_full[:total_unique_videos_dataset]
                    arr_embeds_full_unique = arr_embeds_full[:total_unique_videos_dataset]
                else:
                    text_embeds_full_unique = text_embeds_full
                    vid_embeds_full_unique = vid_embeds_full
                    arr_embeds_full_unique = arr_embeds_full

            else:  # Single GPU case (no all_gather needed for full set, use locally concatenated)
                text_embeds_full_unique = text_embeds_local_concat
                vid_embeds_full_unique = vid_embeds_local_concat
                arr_embeds_full_unique = idx_embeds_local_concat

                # Compute similarity matrix over the FULL UNIQUE dataset embeddings
            sims = sim_matrix(text_embeds_full_unique, vid_embeds_full_unique).detach().cpu().numpy()

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(sims, arr_embeds_full_unique)
                if self.args.rank == 0:
                    self.logger.info(
                        self.verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                                     # Corrected: Call self.verbose
                                     mode=metric_name, args=self.args))
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = self.format_nested_metrics_for_writer(res, mode=metric_name,
                                                                     # Corrected: Call self.format_nested_metrics_for_writer
                                                                     name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        key = key.replace('[', '_').replace(']', '_')
                        self.writer.add_scalar(f'Val_metrics_{dl_idx}/{key}', val, epoch - 1)

                if self.visualizer is not None and self.args.rank == 0:
                    meta_arr_cat_flattened = {}
                    expected_meta_keys = ['narration_id', 'participant_id', 'video_id', 'narration', 'paths',
                                          'raw_captions']
                    for key in expected_meta_keys:
                        if key in self.meta_arr_storage[dl_idx]:
                            if self.n_gpu > 1 and self.args.rank == 0:
                                # Slice meta_arr_cat_flattened to only include unique elements
                                meta_arr_cat_flattened[key] = self.meta_arr_storage[dl_idx][key][
                                                              :total_unique_videos_dataset]
                            else:  # Single GPU or non-rank 0, take all
                                meta_arr_cat_flattened[key] = self.meta_arr_storage[dl_idx][key]
                        else:
                            meta_arr_cat_flattened[key] = []

                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat_flattened, nested_metrics)

            if self.args.rank == 0:
                print("\n--- Top Clip Retrieval Predictions for ALL Queries ---")

                sentence_csv_path = os.path.join(self.config['data_loader']['args']['meta_dir'],
                                                 "EPIC_100_retrieval_test_sentence.csv")
                df_sentence_csv = pd.read_csv(sentence_csv_path)
                all_narration_id_to_query_text = dict(
                    zip(df_sentence_csv['narration_id'], df_sentence_csv['narration']))

                dataset_instance = self.valid_data_loader[dl_idx].dataset
                numerical_idx_to_video_id_map = dataset_instance.numerical_idx_to_video_id

                all_5s_clip_ids_evaluated_numerical = arr_embeds_full_unique.tolist()
                all_5s_clip_ids_evaluated_strings = [
                    numerical_idx_to_video_id_map.get(idx, f"UNKNOWN_VID_{idx}")
                    for idx in all_5s_clip_ids_evaluated_numerical
                ]

                top_k_results_per_query = 5

                for query_row_idx in range(df_sentence_csv.shape[0]):
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
                            clip_num_str = clip_id_string.split('_')[-1]
                            clip_num = int(clip_num_str)
                            start_approx = clip_num * CLIP_DURATION_SECONDS
                            end_approx = (clip_num + 1) * CLIP_DURATION_SECONDS
                            time_str = f"({start_approx:.1f}s - {end_approx:.1f}s)"
                        except ValueError:
                            time_str = "(time N/A)"

                        print(f"    Rank {rank + 1}: {clip_id_string} {time_str} | Score: {score:.4f}")

                print("\n--- End Top Clip Retrieval Predictions for ALL Queries ---")

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

    # --- MODIFIED: Moved these functions inside the class and made them staticmethods ---
    # They are now called as self.verbose(...) and self.format_nested_metrics_for_writer(...)
    # This fixes the NameError in multi-process contexts.
    @staticmethod
    def verbose(epoch, metrics, mode, args, name="TEST"):
        if dist.get_rank() == 0:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            stats_file = open(Path(args.save_dir) / 'stats_vtc.txt', 'a',
                              buffering=1)  # Corrected: args.save_dir (removed extra .args)
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

    @staticmethod
    def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
        res = {}
        for key, val in metrics.items():
            log_name = f"[{mode}]{name}_{key}"
            res[log_name] = val
        return res
# --- END MODIFIED: Moved functions ---