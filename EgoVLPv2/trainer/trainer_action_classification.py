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

class Multi_Trainer_Action_Classification(Multi_BaseTrainer_dist):
    """
    Trainer class for action classification
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
        """
        Training logic for an epoch
        """
        if dist.get_rank() == 0:
            Path(self.args.save_dir).mkdir(parents=True, exist_ok=True)
            stats_file = open(Path(self.args.save_dir) / str('stats.txt'), 'a', buffering=1)
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
                # Move data to GPU
                for k, v in data.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            if torch.is_tensor(vv):
                                data[k][kk] = vv.cuda(gpu, non_blocking=True)
                    elif torch.is_tensor(v):
                        data[k] = v.cuda(gpu, non_blocking=True)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.loss(output, data['label'])

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
                    total = int(self.data_loader[dl_idx].n_samples/self.n_gpu)
                    current = batch_idx * self.data_loader[dl_idx].batch_size
                    final_total = (epoch-1) * total + current
                    self.writer.add_scalar(f'Loss_training/loss_{dl_idx}', loss.detach().item(), final_total)

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.args.print_freq == 0 and self.args.rank == 0:
                    self.logger.info('Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))

                self.optimizer.zero_grad()

            if batch_idx == self.len_epoch:
                break

        log.update({f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))})

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Loss_training/loss_total_{dl_idx}', tl, epoch-1)

        if self.do_validation:
            val_log = self._valid_epoch(epoch, gpu)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch(self, epoch, gpu, save_mis_preds_dir=None, dataset_obj=None):
        """
        Validate after training an epoch. If save_mis_preds_dir is provided, save confusion matrix, mis-predicted stats, and up to 5 mis-predicted videos per action.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        preds = []
        labels = []
        video_ids = []
        true_labels = []
        pred_labels = []
        meta_infos = []
        with torch.no_grad():
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(tqdm(dl)):
                    try:
                        # Move data to GPU
                        for k, v in data.items():
                            if isinstance(v, dict):
                                for kk, vv in v.items():
                                    if torch.is_tensor(vv):
                                        data[k][kk] = vv.cuda(gpu, non_blocking=True)
                            elif torch.is_tensor(v):
                                data[k] = v.cuda(gpu, non_blocking=True)
                        # Forward pass
                        output = self.model(data)
                        loss = self.loss(output, data['label'])
                        # Store predictions and labels
                        preds.append(output.detach().cpu())
                        labels.append(data['label'].detach().cpu())
                        # For confusion matrix and mis-pred tracking
                        batch_pred = output.detach().cpu().argmax(dim=1)
                        batch_true = data['label'].detach().cpu().argmax(dim=1)
                        pred_labels.extend(batch_pred.tolist())
                        true_labels.extend(batch_true.tolist())
                        meta_infos.extend(data['meta'])
                        video_ids.extend([m['paths'] for m in data['meta']])
                        total_val_loss[dl_idx] += loss.item()
                    except Exception as e:
                        print(f"Error processing validation batch {batch_idx} in dataloader {dl_idx}: {str(e)}")
                        print(f"Full traceback:", e.__traceback__)
                        raise
        # Concatenate all predictions and labels
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        # Compute metrics
        metrics = self._eval_metrics((preds, labels))
        for i, metric in enumerate(self.metrics):
            total_val_metrics[0][i] = metrics[i]
        # --- Add upstair+downstair accuracy ---
        try:
            from data_loader.ParkinsonEgo_dataset import ParkinsonEgo
            label_to_idx = ParkinsonEgo.ACTION_LABEL_TO_IDX
            updown_indices = [label_to_idx['upstair'], label_to_idx['downstair']]
            pred_labels_arr = preds.argmax(dim=1)
            updown_total = 0
            updown_correct = 0
            for true_idx, pred_idx in zip(labels.tolist(), pred_labels_arr.tolist()):
                if true_idx in updown_indices:
                    updown_total += 1
                    if true_idx == pred_idx:
                        updown_correct += 1
            updown_acc = updown_correct / updown_total if updown_total > 0 else 0.0
            print(f'Upstair+Downstair accuracy: {updown_acc*100:.2f}% ({updown_correct}/{updown_total})')
        except Exception as e:
            print(f"[Warning] Could not compute upstair+downstair accuracy: {e}")
        # --- End add ---
        # --- Save confusion matrix, mis-predicted stats, and videos if requested ---
        if save_mis_preds_dir is not None and dataset_obj is not None:
            # Confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)
            np.save(os.path.join(save_mis_preds_dir, 'confusion_matrix.npy'), cm)
            # Mis-predicted stats
            mis_pred_counts = {}
            for i in range(dataset_obj.num_classes):
                mis_pred_counts[i] = sum((np.array(true_labels) == i) & (np.array(pred_labels) != i))
            # Save ranked mis-predicted actions
            ranked = sorted(mis_pred_counts.items(), key=lambda x: x[1], reverse=True)
            with open(os.path.join(save_mis_preds_dir, 'mis_pred_stats.txt'), 'w') as f:
                for idx, count in ranked:
                    f.write(f"Action {idx} ({dataset_obj.idx_to_action[idx]}): {count} mis-predicted\n")
            # Save up to 5 mis-predicted videos per action
            mis_pred_examples = {i: [] for i in range(dataset_obj.num_classes)}
            for i, (t, p, vid) in enumerate(zip(true_labels, pred_labels, video_ids)):
                if t != p and len(mis_pred_examples[t]) < 5:
                    mis_pred_examples[t].append((vid, t, p))
            for action_idx, examples in mis_pred_examples.items():
                for j, (vid, t, p) in enumerate(examples):
                    # Reconstruct video path
                    video_fp, _ = dataset_obj._get_video_path({'video_id': vid})
                    dst = os.path.join(save_mis_preds_dir, f'action{action_idx}_mis{j}_true{t}_pred{p}.mp4')
                    try:
                        shutil.copy(video_fp, dst)
                    except Exception as e:
                        print(f"[Warning] Could not copy video {video_fp} to {dst}: {e}")
        # --- End save ---
        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.valid_data_loader)):
                tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                self.writer.add_scalar(f'Loss_val/loss_total_{dl_idx}', tl, epoch-1)
        return {
            'val_loss': total_val_loss[0] / len(self.valid_data_loader[0]),
            'val_metrics': total_val_metrics[0].tolist(),
            'val_updown_acc': updown_acc if 'updown_acc' in locals() else None
        }

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = self.data_loader[dl_idx].n_samples
        else:
            current = batch_idx
            total = len(self.data_loader[dl_idx])
        return base.format(current, total, 100.0 * current / total)

    def train(self, gpu):
        log = super().train(gpu)
        # After training, run test set evaluation (only on main process)
        if self.args.rank == 0:
            from EgoVLPv2.eval_action_classification_test import run_test_evaluation
            import os
            ckpt_dir = self.args.save_dir
            best_ckpt = os.path.join(ckpt_dir, 'model_best.pth')
            if os.path.exists(best_ckpt):
                print(f"\n[INFO] Running test set evaluation with checkpoint: {best_ckpt}")
                test_results = run_test_evaluation(
                    config_path=self.config.config_fname,
                    checkpoint_path=best_ckpt,
                    batch_size=self.batch_size,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    save_csv=os.path.join(ckpt_dir, 'test_predictions.csv'),
                    save_dir=ckpt_dir,
                    resume=None
                )
                print(f"\n[TEST RESULTS] Test set accuracy: {test_results['test_accuracy']*100:.2f}% ({test_results['test_correct']}/{test_results['test_total']})")
                print(f"[TEST RESULTS] Upstair+Downstair accuracy: {test_results['updown_accuracy']*100:.2f}% ({test_results['updown_correct']}/{test_results['updown_total']})")
            else:
                print("[WARNING] No best checkpoint found for test evaluation.")
        return log 