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

from base import BaseTrainer, Multi_BaseTrainer_dist
from model.model import sim_matrix
from utils import inf_loop
from pathlib import Path
import sys
import json
import os
from sklearn.metrics import confusion_matrix
import shutil
import warnings
warnings.filterwarnings("ignore")

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
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )

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
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
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

        if self.do_validation and (epoch == 1 or epoch % 20 == 0):
            val_log = self._valid_epoch(epoch, gpu)
            # Only rank 0 gets validation results, others get empty dict
            if dist.get_rank() == 0:
                val_acc = val_log.get('val_distance_based_accuracy', None)
                if val_acc is None:
                    print(f"Warning: val_acc is None at epoch {epoch}, val_log: {val_log}")
                    val_acc = 0.0  # Set default value instead of crashing
                assert val_acc is not None, f"val_acc is None at epoch {epoch}, val_log: {val_log}"
                stats_file = open(Path(self.args.save_dir) / str('stats_vtc.txt'), 'a', buffering=1)
                print(f"[VAL] Epoch {epoch}: val_acc={val_acc}, best_val_acc={getattr(self, 'best_val_acc', None)}", file=stats_file)
                save_dir = None
                if val_acc is not None and (not hasattr(self, 'best_val_acc') or val_acc > self.best_val_acc):
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    # Save best checkpoint
                    best_ckpt_path = os.path.join(self.args.save_dir, 'model_best.pth')
                    torch.save(self.model.state_dict(), best_ckpt_path)
                    save_dir = os.path.join(self.args.save_dir, 'best_val_ckpt')
                # Save confusion matrix and mis-predicted stats for best epoch
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    self._valid_epoch(epoch, gpu, save_mis_preds_dir=save_dir, dataset_obj=self.valid_data_loader[0].dataset if self.valid_data_loader else None)
                log.update(val_log)
            else:
                # Non-rank 0 processes don't get validation results, so no assertion needed
                val_acc = None

        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics)) if self.metrics else np.zeros(0)
        
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                # The dataset already returns tokenized text, so we don't need to tokenize again
                # Just move the tensors to GPU
                data['text'] = {key: val.cuda(gpu, non_blocking=True) for key, val in data['text'].items()}
                data['video'] = data['video'].cuda(gpu, non_blocking=True)
                data['label'] = data['label'].cuda(gpu, non_blocking=True)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    with torch.cuda.amp.autocast():
                        loss, loss_dict, _ = self.model(data, self.allgather, self.n_gpu, self.args, self.config, self.loss, gpu, task_names='Dual', dataset_name='parkinson')
                        assert loss == loss_dict['Dual']

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()

                if dist.get_rank()==0:
                    if (batch_idx) % self.args.print_freq == 0:
                        stats = dict(epoch=epoch, step=batch_idx,
                                lr_weights=self.optimizer.param_groups[0]['lr'],
                                loss=loss.item())
                        print(json.dumps(stats), file=stats_file)


                if self.writer is not None and self.args.rank == 0:
                    # self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())
                    total = int(self.data_loader[dl_idx].n_samples/self.n_gpu)
                    current = batch_idx * self.data_loader[dl_idx].batch_size
                    final_total = (epoch-1) * total + current
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

            #if batch_idx == 100:
            #    break

            if batch_idx == self.len_epoch:
                break

        log.update({f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))})

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Loss_training/loss_total_{dl_idx}', tl, epoch)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log


    def _valid_epoch(self, epoch, gpu, save_mis_preds_dir=None, dataset_obj=None, return_predictions=False):
        """
        Validate after training an epoch - Distance-based action classification

        :return: A log that contains information about validation
        """
        self.model.eval()
        
        # Pre-compute action text embeddings
        action_texts = ["sitting", "walking", "standing", "upstair", "downstair"]
        action_tokens = self.tokenizer(action_texts, padding=True, truncation=True, return_tensors='pt')
        action_tokens = {k: v.cuda(gpu) for k, v in action_tokens.items()}
        action_embeddings = self.model.module.compute_text(action_tokens)  # [5, embed_dim]
        
        # Evaluation loop
        all_predictions = []
        all_labels = []
        video_ids = []
        pred_label_dicts = []
        
        with torch.no_grad():
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(tqdm(dl)):
                    try:
                        # Move all tensors to GPU
                        data['text'] = {key: val.cuda(gpu, non_blocking=True) for key, val in data['text'].items()}
                        data['video'] = data['video'].cuda(gpu, non_blocking=True)
                        data['label'] = data['label'].cuda(gpu, non_blocking=True)
                        
                        # Get video embeddings
                        ret = self.model.module.infer(data, return_embeds=True, task_names="Dual", ret={})
                        video_embeds = ret['video_embeds']  # [batch_size, embed_dim]
                        
                        # Compute distances between video embeddings and action text embeddings
                        distances = torch.cdist(video_embeds, action_embeddings)  # [batch_size, 5]
                        predictions = torch.argmin(distances, dim=1)  # [batch_size]
                        
                        # Store predictions and labels
                        all_predictions.append(predictions.cpu())
                        all_labels.append(data['label'].cpu())
                        # Robustly extract video IDs from meta
                        meta = data['meta']
                        batch_video_ids = []
                        if isinstance(meta, list):
                            if len(meta) > 0 and isinstance(meta[0], dict) and 'paths' in meta[0]:
                                batch_video_ids = [m['paths'] for m in meta]
                            else:
                                batch_video_ids = [str(m) for m in meta]
                        elif isinstance(meta, dict) and 'paths' in meta:
                            batch_video_ids = meta['paths'] if isinstance(meta['paths'], list) else [meta['paths']]
                        else:
                            batch_video_ids = [str(meta)]
                        # If returning predictions, collect them as dicts
                        if return_predictions:
                            for vp, pred_idx in zip(batch_video_ids, predictions.cpu().tolist()):
                                pred_label_dicts.append({
                                    "video_path": vp,
                                    "classified_action": action_texts[pred_idx]
                                })
                        video_ids.extend(batch_video_ids)
                        
                    except Exception as e:
                        if self.args.rank == 0:
                            print(f"Error processing validation batch {batch_idx} in dataloader {dl_idx}: {str(e)}")
                        continue

        # Concatenate all predictions and labels
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Compute classification accuracy
            correct = (all_predictions == all_labels).sum().item()
            total = all_predictions.size(0)
            accuracy = correct / total if total > 0 else 0.0
            
            # Log results
            if self.args.rank == 0:
                msg = f"[VALIDATION] Epoch {epoch}, Distance-based Classification Accuracy: {accuracy:.4f} ({correct}/{total})"
                print(msg)
                
                if self.writer is not None:
                    self.writer.add_scalar('Val_metrics/distance_based_accuracy', accuracy, epoch)
            
            res_dict = {}
            if self.args.rank == 0:
                res_dict = {
                    'val_distance_based_accuracy': accuracy,
                    'val_correct_predictions': correct,
                    'val_total_predictions': total
                }
            
            # --- Save confusion matrix, mis-predicted stats, and videos if requested ---
            if save_mis_preds_dir is not None and dataset_obj is not None:
                true_labels = all_labels.numpy()
                pred_labels = all_predictions.numpy()
                # Confusion matrix
                cm = confusion_matrix(true_labels, pred_labels)
                np.save(os.path.join(save_mis_preds_dir, 'confusion_matrix.npy'), cm)
                # Mis-predicted stats
                mis_pred_counts = {}
                num_classes = 5  # Hardcoded for now: ["sitting", "walking", "standing", "upstair", "downstair"]
                action_names = ["sitting", "walking", "standing", "upstair", "downstair"]
                for i in range(num_classes):
                    mis_pred_counts[i] = sum((np.array(true_labels) == i) & (np.array(pred_labels) != i))
                # Save ranked mis-predicted actions
                ranked = sorted(mis_pred_counts.items(), key=lambda x: x[1], reverse=True)
                with open(os.path.join(save_mis_preds_dir, 'mis_pred_stats.txt'), 'w') as f:
                    for idx, count in ranked:
                        action_name = action_names[idx] if idx < len(action_names) else f"action_{idx}"
                        f.write(f"Action {idx} ({action_name}): {count} mis-predicted\n")
                # Save up to 5 mis-predicted videos per action
                mis_pred_examples = {i: [] for i in range(num_classes)}
                for i, (t, p, vid) in enumerate(zip(true_labels, pred_labels, video_ids)):
                    if t != p and len(mis_pred_examples[t]) < 5:
                        mis_pred_examples[t].append((vid, t, p))
                for action_idx, examples in mis_pred_examples.items():
                    for j, (vid, t, p) in enumerate(examples):
                        # Reconstruct video path based on annotation logic
                        # vid is like 'video_0_clip_504'
                        try:
                            folder, clip = vid.split('_', 1)  # folder='video_0', clip='clip_504'
                            video_fp = os.path.join(dataset_obj.data_dir, folder, f"{clip}.mp4")
                            dst = os.path.join(save_mis_preds_dir, f'action{action_idx}_mis{j}_true{t}_pred{p}.mp4')
                            shutil.copy(video_fp, dst)
                        except Exception as e:
                            print(f"[Warning] Could not copy video for video_id {vid}: {e}")
            # --- End save ---

            if return_predictions:
                return pred_label_dicts
            else:
                return res_dict
        else:
            # Return empty dict for non-rank 0 processes, or default values for rank 0
            if return_predictions:
                return pred_label_dicts
            elif self.args.rank == 0:
                return {'val_distance_based_accuracy': 0.0, 'val_correct_predictions': 0, 'val_total_predictions': 0}
            else:
                return {}

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

    if dist.get_rank()==0:
        stats = dict(epoch=epoch, msg=msg)
        print(json.dumps(stats), file=stats_file)

    return msg

def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res

def oscc_metrics(preds, labels):
    metrics = {}
    correct = 0
    total = 0
    for pred, label in zip(preds, labels):
        pred_ = torch.argmax(pred)
        if pred_.item() == label.item():
            correct += 1
        total += 1
    accuracy = correct / total
    metrics['accuracy'] = accuracy * 100
    return metrics
